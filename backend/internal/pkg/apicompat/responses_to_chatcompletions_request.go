package apicompat

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ResponsesRequestToChatCompletions converts an OpenAI Responses API request
// into an OpenAI-compatible Chat Completions request.
func ResponsesRequestToChatCompletions(req *ResponsesRequest) (*ChatCompletionsRequest, error) {
	if req == nil {
		return nil, fmt.Errorf("nil responses request")
	}

	messages, err := responsesInputToChatMessages(req.Input)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(req.Instructions) != "" {
		content, _ := json.Marshal(req.Instructions)
		messages = append([]ChatMessage{{Role: "system", Content: content}}, messages...)
	}

	out := &ChatCompletionsRequest{
		Model:           req.Model,
		Messages:        messages,
		Temperature:     req.Temperature,
		TopP:            req.TopP,
		Stream:          req.Stream,
		ToolChoice:      req.ToolChoice,
		ServiceTier:     req.ServiceTier,
		ReasoningEffort: responsesReasoningEffortForChat(req.Reasoning),
	}
	if req.Stream {
		out.StreamOptions = &ChatStreamOptions{IncludeUsage: true}
	}
	if req.MaxOutputTokens != nil && *req.MaxOutputTokens > 0 {
		v := *req.MaxOutputTokens
		out.MaxTokens = &v
	}
	if len(req.Tools) > 0 {
		out.Tools = responsesToolsToChatTools(req.Tools)
	}

	return out, nil
}

func responsesReasoningEffortForChat(reasoning *ResponsesReasoning) string {
	if reasoning == nil {
		return ""
	}
	effort := strings.TrimSpace(reasoning.Effort)
	if effort == "none" {
		return ""
	}
	return effort
}

func responsesToolsToChatTools(tools []ResponsesTool) []ChatTool {
	out := make([]ChatTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "function" || strings.TrimSpace(tool.Name) == "" {
			continue
		}
		out = append(out, ChatTool{
			Type: "function",
			Function: &ChatFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.Parameters,
				Strict:      tool.Strict,
			},
		})
	}
	return out
}

func responsesInputToChatMessages(raw json.RawMessage) ([]ChatMessage, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	var inputText string
	if err := json.Unmarshal(raw, &inputText); err == nil {
		content, _ := json.Marshal(inputText)
		return []ChatMessage{{Role: "user", Content: content}}, nil
	}

	var items []ResponsesInputItem
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, fmt.Errorf("parse responses input: %w", err)
	}

	messages := make([]ChatMessage, 0, len(items))
	for _, item := range items {
		msgs, err := responsesInputItemToChatMessages(item)
		if err != nil {
			return nil, err
		}
		messages = append(messages, msgs...)
	}
	return messages, nil
}

func responsesInputItemToChatMessages(item ResponsesInputItem) ([]ChatMessage, error) {
	switch item.Type {
	case "function_call":
		callID := strings.TrimSpace(item.CallID)
		if callID == "" {
			callID = strings.TrimSpace(item.ID)
		}
		args := item.Arguments
		if strings.TrimSpace(args) == "" {
			args = "{}"
		}
		return []ChatMessage{{
			Role: "assistant",
			ToolCalls: []ChatToolCall{{
				ID:   callID,
				Type: "function",
				Function: ChatFunctionCall{
					Name:      item.Name,
					Arguments: args,
				},
			}},
		}}, nil
	case "function_call_output":
		output := item.Output
		if output == "" {
			output = "(empty)"
		}
		content, _ := json.Marshal(output)
		return []ChatMessage{{
			Role:       "tool",
			ToolCallID: item.CallID,
			Content:    content,
		}}, nil
	}

	role := strings.TrimSpace(item.Role)
	if role == "" && item.Type == "message" {
		role = "user"
	}
	switch role {
	case "developer":
		role = "system"
	case "system", "user", "assistant", "tool":
	default:
		role = "user"
	}

	content, err := responsesContentToChatContent(item.Content, role)
	if err != nil {
		return nil, err
	}
	return []ChatMessage{{Role: role, Content: content, ToolCallID: item.CallID}}, nil
}

func responsesContentToChatContent(raw json.RawMessage, role string) (json.RawMessage, error) {
	if len(raw) == 0 {
		if role == "assistant" {
			return nil, nil
		}
		return json.Marshal("")
	}

	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return json.Marshal(s)
	}

	var parts []ResponsesContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return raw, nil
	}

	chatParts := make([]ChatContentPart, 0, len(parts))
	hasImage := false
	for _, part := range parts {
		switch part.Type {
		case "input_text", "output_text", "text":
			if part.Text != "" {
				chatParts = append(chatParts, ChatContentPart{Type: "text", Text: part.Text})
			}
		case "input_image", "image_url":
			if part.ImageURL != "" {
				hasImage = true
				chatParts = append(chatParts, ChatContentPart{
					Type:     "image_url",
					ImageURL: &ChatImageURL{URL: part.ImageURL},
				})
			}
		}
	}
	if len(chatParts) == 0 {
		return json.Marshal("")
	}
	if !hasImage || role == "assistant" || role == "system" || role == "tool" {
		return json.Marshal(joinChatTextParts(chatParts))
	}
	return json.Marshal(chatParts)
}

func joinChatTextParts(parts []ChatContentPart) string {
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n\n")
}
