package apicompat

import (
	"encoding/json"
	"strings"
)

// ChatCompletionsToResponsesResponse converts a non-streaming Chat Completions
// response into an OpenAI Responses API response.
func ChatCompletionsToResponsesResponse(resp *ChatCompletionsResponse, model string) *ResponsesResponse {
	if resp == nil {
		return &ResponsesResponse{
			ID:     generateResponsesID(),
			Object: "response",
			Model:  model,
			Status: "completed",
			Output: []ResponsesOutput{},
		}
	}
	if strings.TrimSpace(model) == "" {
		model = resp.Model
	}
	id := resp.ID
	if strings.TrimSpace(id) == "" {
		id = generateResponsesID()
	}

	status := "completed"
	var incompleteDetails *ResponsesIncompleteDetails
	var output []ResponsesOutput
	if len(resp.Choices) > 0 {
		choice := resp.Choices[0]
		status, incompleteDetails = chatFinishReasonToResponsesStatus(choice.FinishReason)
		output = chatMessageToResponsesOutput(choice.Message)
	}
	if output == nil {
		output = []ResponsesOutput{}
	}

	return &ResponsesResponse{
		ID:                id,
		Object:            "response",
		Model:             model,
		Status:            status,
		Output:            output,
		Usage:             chatUsageToResponsesUsage(resp.Usage),
		IncompleteDetails: incompleteDetails,
	}
}

func chatFinishReasonToResponsesStatus(finishReason string) (string, *ResponsesIncompleteDetails) {
	switch finishReason {
	case "length":
		return "incomplete", &ResponsesIncompleteDetails{Reason: "max_output_tokens"}
	case "content_filter":
		return "incomplete", &ResponsesIncompleteDetails{Reason: "content_filter"}
	default:
		return "completed", nil
	}
}

func chatMessageToResponsesOutput(msg ChatMessage) []ResponsesOutput {
	var output []ResponsesOutput
	if strings.TrimSpace(msg.ReasoningContent) != "" {
		output = append(output, ResponsesOutput{
			Type: "reasoning",
			Summary: []ResponsesSummary{{
				Type: "summary_text",
				Text: msg.ReasoningContent,
			}},
		})
	}

	text := chatContentText(msg.Content)
	if text != "" || len(msg.ToolCalls) == 0 {
		output = append(output, ResponsesOutput{
			Type:   "message",
			ID:     generateResponsesMessageID(),
			Role:   "assistant",
			Status: "completed",
			Content: []ResponsesContentPart{{
				Type: "output_text",
				Text: text,
			}},
		})
	}

	for _, tc := range msg.ToolCalls {
		output = append(output, ResponsesOutput{
			Type:      "function_call",
			CallID:    tc.ID,
			Name:      tc.Function.Name,
			Arguments: normalizeChatToolArguments(tc.Function.Arguments),
		})
	}
	return output
}

func chatContentText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []ChatContentPart
	if err := json.Unmarshal(raw, &parts); err == nil {
		texts := make([]string, 0, len(parts))
		for _, part := range parts {
			if part.Type == "text" && part.Text != "" {
				texts = append(texts, part.Text)
			}
		}
		return strings.Join(texts, "\n\n")
	}
	return ""
}

func chatUsageToResponsesUsage(usage *ChatUsage) *ResponsesUsage {
	if usage == nil {
		return nil
	}
	out := &ResponsesUsage{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
		TotalTokens:  usage.TotalTokens,
	}
	if out.TotalTokens == 0 {
		out.TotalTokens = out.InputTokens + out.OutputTokens
	}
	if usage.PromptTokensDetails != nil && usage.PromptTokensDetails.CachedTokens > 0 {
		out.InputTokensDetails = &ResponsesInputTokensDetails{
			CachedTokens: usage.PromptTokensDetails.CachedTokens,
		}
	}
	return out
}

func normalizeChatToolArguments(args string) string {
	if strings.TrimSpace(args) == "" {
		return "{}"
	}
	return args
}

func generateResponsesMessageID() string {
	id := generateResponsesID()
	return strings.Replace(id, "resp_", "msg_", 1)
}

type chatResponsesToolState struct {
	OutputIndex int
	CallID      string
	Name        string
	Arguments   strings.Builder
}

// ChatEventToResponsesState tracks state while converting Chat Completions SSE
// chunks into Responses SSE events.
type ChatEventToResponsesState struct {
	ResponseID     string
	Model          string
	SequenceNumber int
	CreatedSent    bool
	CompletedSent  bool

	MessageOutputIndex int
	MessageItemID      string
	MessageItemAdded   bool
	Text               strings.Builder
	Reasoning          strings.Builder

	NextOutputIndex int
	ToolCalls       map[int]*chatResponsesToolState
	ToolOrder       []int

	Usage        *ResponsesUsage
	FinishReason string
}

func NewChatEventToResponsesState() *ChatEventToResponsesState {
	return &ChatEventToResponsesState{
		MessageOutputIndex: -1,
		ToolCalls:          make(map[int]*chatResponsesToolState),
	}
}

// ChatChunkToResponsesEvents converts one Chat Completions streaming chunk into
// zero or more Responses streaming events.
func ChatChunkToResponsesEvents(chunk *ChatCompletionsChunk, state *ChatEventToResponsesState) []ResponsesStreamEvent {
	if state == nil {
		state = NewChatEventToResponsesState()
	}
	if chunk == nil {
		return nil
	}
	if strings.TrimSpace(state.ResponseID) == "" {
		state.ResponseID = strings.TrimSpace(chunk.ID)
	}
	if strings.TrimSpace(state.ResponseID) == "" {
		state.ResponseID = generateResponsesID()
	}
	if strings.TrimSpace(state.Model) == "" {
		state.Model = strings.TrimSpace(chunk.Model)
	}
	if chunk.Usage != nil {
		state.Usage = chatUsageToResponsesUsage(chunk.Usage)
	}

	var events []ResponsesStreamEvent
	if !state.CreatedSent {
		events = append(events, chatResponsesCreatedEvent(state))
		state.CreatedSent = true
	}

	for _, choice := range chunk.Choices {
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			state.FinishReason = *choice.FinishReason
		}
		if choice.Delta.ReasoningContent != nil && *choice.Delta.ReasoningContent != "" {
			_, _ = state.Reasoning.WriteString(*choice.Delta.ReasoningContent)
		}
		if choice.Delta.Content != nil {
			events = append(events, chatResponsesTextDeltaEvents(state, *choice.Delta.Content)...)
		}
		for _, tc := range choice.Delta.ToolCalls {
			events = append(events, chatResponsesToolCallEvents(state, tc)...)
		}
	}

	return events
}

// FinalizeChatResponsesStream emits Responses completion events for a Chat
// Completions stream. It is idempotent.
func FinalizeChatResponsesStream(state *ChatEventToResponsesState) []ResponsesStreamEvent {
	if state == nil || state.CompletedSent {
		return nil
	}
	if strings.TrimSpace(state.ResponseID) == "" {
		state.ResponseID = generateResponsesID()
	}
	if strings.TrimSpace(state.Model) == "" {
		state.Model = "unknown"
	}

	var events []ResponsesStreamEvent
	if !state.CreatedSent {
		events = append(events, chatResponsesCreatedEvent(state))
		state.CreatedSent = true
	}
	if state.MessageItemAdded {
		events = append(events, chatResponsesEvent(state, "response.output_text.done", &ResponsesStreamEvent{
			OutputIndex:  state.MessageOutputIndex,
			ContentIndex: 0,
			ItemID:       state.MessageItemID,
			Text:         state.Text.String(),
		}))
		events = append(events, chatResponsesEvent(state, "response.output_item.done", &ResponsesStreamEvent{
			OutputIndex: state.MessageOutputIndex,
			Item: &ResponsesOutput{
				Type:   "message",
				ID:     state.MessageItemID,
				Role:   "assistant",
				Status: "completed",
				Content: []ResponsesContentPart{{
					Type: "output_text",
					Text: state.Text.String(),
				}},
			},
		}))
	}
	for _, idx := range state.ToolOrder {
		tool := state.ToolCalls[idx]
		if tool == nil {
			continue
		}
		args := normalizeChatToolArguments(tool.Arguments.String())
		events = append(events, chatResponsesEvent(state, "response.function_call_arguments.done", &ResponsesStreamEvent{
			OutputIndex: tool.OutputIndex,
			CallID:      tool.CallID,
			Name:        tool.Name,
			Arguments:   args,
		}))
		events = append(events, chatResponsesEvent(state, "response.output_item.done", &ResponsesStreamEvent{
			OutputIndex: tool.OutputIndex,
			Item: &ResponsesOutput{
				Type:      "function_call",
				CallID:    tool.CallID,
				Name:      tool.Name,
				Arguments: args,
			},
		}))
	}

	status, incompleteDetails := chatFinishReasonToResponsesStatus(state.FinishReason)
	events = append(events, chatResponsesEvent(state, "response.completed", &ResponsesStreamEvent{
		Response: &ResponsesResponse{
			ID:                state.ResponseID,
			Object:            "response",
			Model:             state.Model,
			Status:            status,
			Output:            chatResponsesStateOutput(state),
			Usage:             state.Usage,
			IncompleteDetails: incompleteDetails,
		},
	}))
	state.CompletedSent = true
	return events
}

func chatResponsesTextDeltaEvents(state *ChatEventToResponsesState, delta string) []ResponsesStreamEvent {
	var events []ResponsesStreamEvent
	if !state.MessageItemAdded {
		state.MessageOutputIndex = state.NextOutputIndex
		state.NextOutputIndex++
		state.MessageItemID = generateResponsesMessageID()
		state.MessageItemAdded = true
		events = append(events, chatResponsesEvent(state, "response.output_item.added", &ResponsesStreamEvent{
			OutputIndex: state.MessageOutputIndex,
			Item: &ResponsesOutput{
				Type:   "message",
				ID:     state.MessageItemID,
				Role:   "assistant",
				Status: "in_progress",
				Content: []ResponsesContentPart{{
					Type: "output_text",
					Text: "",
				}},
			},
		}))
	}
	_, _ = state.Text.WriteString(delta)
	events = append(events, chatResponsesEvent(state, "response.output_text.delta", &ResponsesStreamEvent{
		OutputIndex:  state.MessageOutputIndex,
		ContentIndex: 0,
		ItemID:       state.MessageItemID,
		Delta:        delta,
	}))
	return events
}

func chatResponsesToolCallEvents(state *ChatEventToResponsesState, tc ChatToolCall) []ResponsesStreamEvent {
	idx := len(state.ToolOrder)
	if tc.Index != nil {
		idx = *tc.Index
	}
	tool := state.ToolCalls[idx]
	var events []ResponsesStreamEvent
	if tool == nil {
		callID := strings.TrimSpace(tc.ID)
		if callID == "" {
			callID = "call_" + strings.TrimPrefix(generateResponsesID(), "resp_")
		}
		tool = &chatResponsesToolState{
			OutputIndex: state.NextOutputIndex,
			CallID:      callID,
			Name:        tc.Function.Name,
		}
		state.NextOutputIndex++
		state.ToolCalls[idx] = tool
		state.ToolOrder = append(state.ToolOrder, idx)
		events = append(events, chatResponsesEvent(state, "response.output_item.added", &ResponsesStreamEvent{
			OutputIndex: tool.OutputIndex,
			Item: &ResponsesOutput{
				Type:   "function_call",
				CallID: tool.CallID,
				Name:   tool.Name,
			},
		}))
	}
	if strings.TrimSpace(tc.ID) != "" {
		tool.CallID = tc.ID
	}
	if strings.TrimSpace(tc.Function.Name) != "" {
		tool.Name = tc.Function.Name
	}
	if tc.Function.Arguments != "" {
		_, _ = tool.Arguments.WriteString(tc.Function.Arguments)
		events = append(events, chatResponsesEvent(state, "response.function_call_arguments.delta", &ResponsesStreamEvent{
			OutputIndex: tool.OutputIndex,
			CallID:      tool.CallID,
			Name:        tool.Name,
			Delta:       tc.Function.Arguments,
		}))
	}
	return events
}

func chatResponsesStateOutput(state *ChatEventToResponsesState) []ResponsesOutput {
	var output []ResponsesOutput
	if state.Reasoning.Len() > 0 {
		output = append(output, ResponsesOutput{
			Type: "reasoning",
			Summary: []ResponsesSummary{{
				Type: "summary_text",
				Text: state.Reasoning.String(),
			}},
		})
	}
	if state.MessageItemAdded || len(state.ToolOrder) == 0 {
		if state.MessageItemID == "" {
			state.MessageItemID = generateResponsesMessageID()
		}
		output = append(output, ResponsesOutput{
			Type:   "message",
			ID:     state.MessageItemID,
			Role:   "assistant",
			Status: "completed",
			Content: []ResponsesContentPart{{
				Type: "output_text",
				Text: state.Text.String(),
			}},
		})
	}
	for _, idx := range state.ToolOrder {
		tool := state.ToolCalls[idx]
		if tool == nil {
			continue
		}
		output = append(output, ResponsesOutput{
			Type:      "function_call",
			CallID:    tool.CallID,
			Name:      tool.Name,
			Arguments: normalizeChatToolArguments(tool.Arguments.String()),
		})
	}
	return output
}

func chatResponsesCreatedEvent(state *ChatEventToResponsesState) ResponsesStreamEvent {
	return chatResponsesEvent(state, "response.created", &ResponsesStreamEvent{
		Response: &ResponsesResponse{
			ID:     state.ResponseID,
			Object: "response",
			Model:  state.Model,
			Status: "in_progress",
			Output: []ResponsesOutput{},
		},
	})
}

func chatResponsesEvent(state *ChatEventToResponsesState, eventType string, template *ResponsesStreamEvent) ResponsesStreamEvent {
	seq := state.SequenceNumber
	state.SequenceNumber++

	evt := *template
	evt.Type = eventType
	evt.SequenceNumber = seq
	return evt
}
