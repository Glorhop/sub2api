package service

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/util/responseheaders"
	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func (s *OpenAIGatewayService) forwardOpenAIChatCompletionsUpstream(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	token string,
	reqStream bool,
	originalModel string,
	upstreamModel string,
	startTime time.Time,
	reasoningEffort *string,
	serviceTier *string,
) (*OpenAIForwardResult, error) {
	var responsesReq apicompat.ResponsesRequest
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": gin.H{
				"type":    "invalid_request_error",
				"message": "Invalid Responses request body",
			},
		})
		return nil, fmt.Errorf("parse responses request for chat completions upstream: %w", err)
	}
	responsesReq.Stream = reqStream

	chatReq, err := apicompat.ResponsesRequestToChatCompletions(&responsesReq)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{
			"error": gin.H{
				"type":    "invalid_request_error",
				"message": "Unable to convert Responses request to Chat Completions",
			},
		})
		return nil, err
	}
	chatReq.Stream = reqStream
	if reqStream {
		chatReq.StreamOptions = &apicompat.ChatStreamOptions{IncludeUsage: true}
	}

	chatBody, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("serialize chat completions request: %w", err)
	}
	if c != nil {
		c.Set("_gateway_upstream_endpoint_override", "/v1/chat/completions")
	}
	setOpsUpstreamRequestBody(c, chatBody)

	upstreamCtx, releaseUpstreamCtx := detachStreamUpstreamContext(ctx, reqStream)
	upstreamReq, err := s.buildOpenAIChatCompletionsUpstreamRequest(upstreamCtx, c, account, chatBody, token)
	releaseUpstreamCtx()
	if err != nil {
		return nil, err
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: 0,
			Kind:               "request_error",
			Message:            safeErr,
		})
		c.JSON(http.StatusBadGateway, gin.H{
			"error": gin.H{
				"type":    "upstream_error",
				"message": "Upstream request failed",
			},
		})
		return nil, fmt.Errorf("chat completions upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		upstreamMsg := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractUpstreamErrorMessage(respBody)))
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, upstreamMsg, respBody) {
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  resp.Header.Get("x-request-id"),
				Kind:               "failover",
				Message:            upstreamMsg,
			})
			s.handleFailoverSideEffects(ctx, resp, account)
			return nil, &UpstreamFailoverError{
				StatusCode:             resp.StatusCode,
				ResponseBody:           respBody,
				RetryableOnSameAccount: account.IsPoolMode() && (isPoolModeRetryableStatus(resp.StatusCode) || isOpenAITransientProcessingError(resp.StatusCode, upstreamMsg, respBody)),
			}
		}
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		return s.handleErrorResponse(ctx, resp, c, account, chatBody)
	}

	var usage *OpenAIUsage
	var firstTokenMs *int
	if reqStream {
		streamResult, err := s.handleChatCompletionsUpstreamStreamingResponse(ctx, resp, c, startTime, originalModel)
		if err != nil {
			return nil, err
		}
		usage = streamResult.usage
		firstTokenMs = streamResult.firstTokenMs
	} else {
		usage, err = s.handleChatCompletionsUpstreamNonStreamingResponse(resp, c, originalModel)
		if err != nil {
			return nil, err
		}
	}
	if usage == nil {
		usage = &OpenAIUsage{}
	}

	return &OpenAIForwardResult{
		RequestID:       resp.Header.Get("x-request-id"),
		Usage:           *usage,
		Model:           originalModel,
		UpstreamModel:   upstreamModel,
		ServiceTier:     serviceTier,
		ReasoningEffort: reasoningEffort,
		Stream:          reqStream,
		OpenAIWSMode:    false,
		Duration:        time.Since(startTime),
		FirstTokenMs:    firstTokenMs,
	}, nil
}

func (s *OpenAIGatewayService) forwardOpenAIChatCompletionsUpstreamAsAnthropic(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	token string,
	clientStream bool,
	originalModel string,
	billingModel string,
	upstreamModel string,
	startTime time.Time,
	promptCacheKey string,
) (*OpenAIForwardResult, error) {
	var responsesReq apicompat.ResponsesRequest
	if err := json.Unmarshal(body, &responsesReq); err != nil {
		writeAnthropicError(c, http.StatusBadRequest, "invalid_request_error", "Invalid Responses request body")
		return nil, fmt.Errorf("parse responses request for chat completions upstream: %w", err)
	}
	responsesReq.Stream = clientStream

	chatReq, err := apicompat.ResponsesRequestToChatCompletions(&responsesReq)
	if err != nil {
		writeAnthropicError(c, http.StatusBadRequest, "invalid_request_error", "Unable to convert request to Chat Completions")
		return nil, err
	}
	chatReq.Stream = clientStream
	if clientStream {
		chatReq.StreamOptions = &apicompat.ChatStreamOptions{IncludeUsage: true}
	}

	chatBody, err := json.Marshal(chatReq)
	if err != nil {
		return nil, fmt.Errorf("serialize chat completions request: %w", err)
	}
	if c != nil {
		c.Set("_gateway_upstream_endpoint_override", "/v1/chat/completions")
	}
	setOpsUpstreamRequestBody(c, chatBody)

	upstreamCtx, releaseUpstreamCtx := detachStreamUpstreamContext(ctx, clientStream)
	upstreamReq, err := s.buildOpenAIChatCompletionsUpstreamRequest(upstreamCtx, c, account, chatBody, token)
	releaseUpstreamCtx()
	if err != nil {
		return nil, err
	}
	if promptCacheKey != "" {
		apiKeyID := getAPIKeyIDFromContext(c)
		upstreamReq.Header.Set("session_id", generateSessionUUID(isolateOpenAISessionID(apiKeyID, promptCacheKey)))
	}

	proxyURL := ""
	if account.ProxyID != nil && account.Proxy != nil {
		proxyURL = account.Proxy.URL()
	}

	upstreamStart := time.Now()
	resp, err := s.httpUpstream.Do(upstreamReq, proxyURL, account.ID, account.Concurrency)
	SetOpsLatencyMs(c, OpsUpstreamLatencyMsKey, time.Since(upstreamStart).Milliseconds())
	if err != nil {
		safeErr := sanitizeUpstreamErrorMessage(err.Error())
		setOpsUpstreamError(c, 0, safeErr, "")
		appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
			Platform:           account.Platform,
			AccountID:          account.ID,
			AccountName:        account.Name,
			UpstreamStatusCode: 0,
			Kind:               "request_error",
			Message:            safeErr,
		})
		writeAnthropicError(c, http.StatusBadGateway, "api_error", "Upstream request failed")
		return nil, fmt.Errorf("chat completions upstream request failed: %s", safeErr)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode >= 400 {
		respBody, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<20))
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		upstreamMsg := sanitizeUpstreamErrorMessage(strings.TrimSpace(extractUpstreamErrorMessage(respBody)))
		if s.shouldFailoverOpenAIUpstreamResponse(resp.StatusCode, upstreamMsg, respBody) {
			appendOpsUpstreamError(c, OpsUpstreamErrorEvent{
				Platform:           account.Platform,
				AccountID:          account.ID,
				AccountName:        account.Name,
				UpstreamStatusCode: resp.StatusCode,
				UpstreamRequestID:  resp.Header.Get("x-request-id"),
				Kind:               "failover",
				Message:            upstreamMsg,
			})
			s.handleFailoverSideEffects(ctx, resp, account)
			return nil, &UpstreamFailoverError{
				StatusCode:             resp.StatusCode,
				ResponseBody:           respBody,
				RetryableOnSameAccount: account.IsPoolMode() && (isPoolModeRetryableStatus(resp.StatusCode) || isOpenAITransientProcessingError(resp.StatusCode, upstreamMsg, respBody)),
			}
		}
		resp.Body = io.NopCloser(bytes.NewReader(respBody))
		return s.handleAnthropicErrorResponse(resp, c, account)
	}

	var result *OpenAIForwardResult
	if clientStream {
		result, err = s.handleChatCompletionsUpstreamAnthropicStreamingResponse(ctx, resp, c, startTime, originalModel, billingModel, upstreamModel)
	} else {
		result, err = s.handleChatCompletionsUpstreamAnthropicNonStreamingResponse(resp, c, startTime, originalModel, billingModel, upstreamModel)
	}
	if err != nil {
		return nil, err
	}
	if result != nil {
		if responsesReq.ServiceTier != "" {
			st := responsesReq.ServiceTier
			result.ServiceTier = &st
		}
		if responsesReq.Reasoning != nil && responsesReq.Reasoning.Effort != "" {
			re := responsesReq.Reasoning.Effort
			result.ReasoningEffort = &re
		}
	}
	return result, nil
}

func (s *OpenAIGatewayService) buildOpenAIChatCompletionsUpstreamRequest(
	ctx context.Context,
	c *gin.Context,
	account *Account,
	body []byte,
	token string,
) (*http.Request, error) {
	baseURL := account.GetOpenAIBaseURL()
	if baseURL == "" {
		baseURL = "https://api.openai.com"
	}
	validatedURL, err := s.validateUpstreamBaseURL(baseURL)
	if err != nil {
		return nil, err
	}
	targetURL := buildOpenAIChatCompletionsURL(validatedURL)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("authorization", "Bearer "+token)
	req.Header.Set("content-type", "application/json")

	if c != nil && c.Request != nil {
		for key, values := range c.Request.Header {
			lowerKey := strings.ToLower(key)
			if openaiAllowedHeaders[lowerKey] {
				for _, v := range values {
					req.Header.Add(key, v)
				}
			}
		}
	}
	req.Header.Set("content-type", "application/json")
	return req, nil
}

func (s *OpenAIGatewayService) handleChatCompletionsUpstreamNonStreamingResponse(
	resp *http.Response,
	c *gin.Context,
	originalModel string,
) (*OpenAIUsage, error) {
	body, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, openAITooLargeError)
	if err != nil {
		return nil, err
	}
	if isEventStreamResponse(resp.Header) {
		return nil, errors.New("chat completions upstream returned SSE for a non-stream request")
	}

	var chatResp apicompat.ChatCompletionsResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		return nil, fmt.Errorf("parse chat completions response: %w", err)
	}
	responsesResp := apicompat.ChatCompletionsToResponsesResponse(&chatResp, originalModel)
	out, err := json.Marshal(responsesResp)
	if err != nil {
		return nil, fmt.Errorf("serialize responses response: %w", err)
	}

	responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	c.Data(resp.StatusCode, "application/json", out)

	return openAIUsageFromResponsesUsage(responsesResp.Usage), nil
}

func (s *OpenAIGatewayService) handleChatCompletionsUpstreamAnthropicNonStreamingResponse(
	resp *http.Response,
	c *gin.Context,
	startTime time.Time,
	originalModel string,
	billingModel string,
	upstreamModel string,
) (*OpenAIForwardResult, error) {
	body, err := ReadUpstreamResponseBody(resp.Body, s.cfg, c, anthropicTooLargeError)
	if err != nil {
		return nil, err
	}
	if isEventStreamResponse(resp.Header) {
		writeAnthropicError(c, http.StatusBadGateway, "api_error", "Chat Completions upstream returned SSE for a non-stream request")
		return nil, errors.New("chat completions upstream returned SSE for a non-stream request")
	}

	var chatResp apicompat.ChatCompletionsResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		writeAnthropicError(c, http.StatusBadGateway, "api_error", "Unable to parse Chat Completions upstream response")
		return nil, fmt.Errorf("parse chat completions response: %w", err)
	}
	responsesResp := apicompat.ChatCompletionsToResponsesResponse(&chatResp, upstreamModel)
	anthropicResp := apicompat.ResponsesToAnthropic(responsesResp, originalModel)
	out, err := json.Marshal(anthropicResp)
	if err != nil {
		return nil, fmt.Errorf("serialize anthropic response: %w", err)
	}

	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Data(resp.StatusCode, "application/json", out)

	usage := openAIUsageFromResponsesUsage(responsesResp.Usage)
	if usage == nil {
		usage = &OpenAIUsage{}
	}
	return &OpenAIForwardResult{
		RequestID:     resp.Header.Get("x-request-id"),
		Usage:         *usage,
		Model:         originalModel,
		BillingModel:  billingModel,
		UpstreamModel: upstreamModel,
		Stream:        false,
		Duration:      time.Since(startTime),
	}, nil
}

func (s *OpenAIGatewayService) handleChatCompletionsUpstreamAnthropicStreamingResponse(
	ctx context.Context,
	resp *http.Response,
	c *gin.Context,
	startTime time.Time,
	originalModel string,
	billingModel string,
	upstreamModel string,
) (*OpenAIForwardResult, error) {
	requestID := resp.Header.Get("x-request-id")
	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("X-Accel-Buffering", "no")
	if requestID != "" {
		c.Writer.Header().Set("x-request-id", requestID)
	}
	c.Writer.WriteHeader(http.StatusOK)

	w := c.Writer
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	chatState := apicompat.NewChatEventToResponsesState()
	chatState.Model = upstreamModel
	anthropicState := apicompat.NewResponsesEventToAnthropicState()
	anthropicState.Model = originalModel

	usage := &OpenAIUsage{}
	var firstTokenMs *int
	seenClientOutput := false

	resultWithUsage := func() *OpenAIForwardResult {
		return &OpenAIForwardResult{
			RequestID:     requestID,
			Usage:         *usage,
			Model:         originalModel,
			BillingModel:  billingModel,
			UpstreamModel: upstreamModel,
			Stream:        true,
			Duration:      time.Since(startTime),
			FirstTokenMs:  firstTokenMs,
		}
	}

	writeAnthropicEvents := func(events []apicompat.AnthropicStreamEvent) error {
		if len(events) == 0 {
			return nil
		}
		for _, evt := range events {
			sse, err := apicompat.ResponsesAnthropicEventToSSE(evt)
			if err != nil {
				logger.L().Warn("chat completions anthropic stream: failed to marshal event",
					zap.Error(err),
					zap.String("request_id", requestID),
				)
				continue
			}
			if _, err := fmt.Fprint(w, sse); err != nil {
				return err
			}
		}
		flusher.Flush()
		return nil
	}

	writeResponsesEvents := func(events []apicompat.ResponsesStreamEvent) error {
		if len(events) == 0 {
			return nil
		}
		for _, evt := range events {
			if !seenClientOutput && chatResponsesEventStartsClientOutput(evt.Type) {
				seenClientOutput = true
				ms := int(time.Since(startTime).Milliseconds())
				firstTokenMs = &ms
			}
			anthropicEvents := apicompat.ResponsesEventToAnthropicEvents(&evt, anthropicState)
			if err := writeAnthropicEvents(anthropicEvents); err != nil {
				return err
			}
		}
		return nil
	}

	finalize := func() (*OpenAIForwardResult, error) {
		events := apicompat.FinalizeChatResponsesStream(chatState)
		if chatState.Usage != nil {
			usage = openAIUsageFromResponsesUsage(chatState.Usage)
		}
		if err := writeResponsesEvents(events); err != nil {
			return resultWithUsage(), nil
		}
		if err := writeAnthropicEvents(apicompat.FinalizeResponsesAnthropicStream(anthropicState)); err != nil {
			return resultWithUsage(), nil
		}
		return resultWithUsage(), nil
	}

	scanner := bufio.NewScanner(resp.Body)
	maxLineSize := defaultMaxLineSize
	if s.cfg != nil && s.cfg.Gateway.MaxLineSize > 0 {
		maxLineSize = s.cfg.Gateway.MaxLineSize
	}
	scanBuf := getSSEScannerBuf64K()
	scanner.Buffer(scanBuf[:0], maxLineSize)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			return finalize()
		}
		var chunk apicompat.ChatCompletionsChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			logger.L().Warn("chat completions anthropic stream: failed to parse stream chunk",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
			continue
		}
		events := apicompat.ChatChunkToResponsesEvents(&chunk, chatState)
		if chatState.Usage != nil {
			usage = openAIUsageFromResponsesUsage(chatState.Usage)
		}
		if err := writeResponsesEvents(events); err != nil {
			logger.L().Info("chat completions anthropic stream: client disconnected",
				zap.String("request_id", requestID),
			)
			return resultWithUsage(), nil
		}
	}
	if err := scanner.Err(); err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			logger.L().Warn("chat completions anthropic stream: read error",
				zap.Error(err),
				zap.String("request_id", requestID),
			)
		}
	}
	return finalize()
}

func (s *OpenAIGatewayService) handleChatCompletionsUpstreamStreamingResponse(
	ctx context.Context,
	resp *http.Response,
	c *gin.Context,
	startTime time.Time,
	originalModel string,
) (*openaiStreamingResult, error) {
	if s.responseHeaderFilter != nil {
		responseheaders.WriteFilteredHeaders(c.Writer.Header(), resp.Header, s.responseHeaderFilter)
	}
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Accel-Buffering", "no")
	if v := resp.Header.Get("x-request-id"); v != "" {
		c.Header("x-request-id", v)
	}

	w := c.Writer
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, errors.New("streaming not supported")
	}

	state := apicompat.NewChatEventToResponsesState()
	state.Model = originalModel
	usage := &OpenAIUsage{}
	var firstTokenMs *int
	seenClientOutput := false

	writeEvents := func(events []apicompat.ResponsesStreamEvent) error {
		if len(events) == 0 {
			return nil
		}
		for _, evt := range events {
			if !seenClientOutput && chatResponsesEventStartsClientOutput(evt.Type) {
				seenClientOutput = true
				ms := int(time.Since(startTime).Milliseconds())
				firstTokenMs = &ms
			}
			sse, err := apicompat.ResponsesEventToSSE(evt)
			if err != nil {
				logger.L().Warn("chat completions upstream: failed to marshal responses event", zap.Error(err))
				continue
			}
			if _, err := fmt.Fprint(w, sse); err != nil {
				return err
			}
		}
		flusher.Flush()
		return nil
	}

	finalize := func() (*openaiStreamingResult, error) {
		events := apicompat.FinalizeChatResponsesStream(state)
		if state.Usage != nil {
			usage = openAIUsageFromResponsesUsage(state.Usage)
		}
		if err := writeEvents(events); err != nil {
			return nil, err
		}
		return &openaiStreamingResult{usage: usage, firstTokenMs: firstTokenMs}, nil
	}

	scanner := bufio.NewScanner(resp.Body)
	maxLineSize := defaultMaxLineSize
	if s.cfg != nil && s.cfg.Gateway.MaxLineSize > 0 {
		maxLineSize = s.cfg.Gateway.MaxLineSize
	}
	scanBuf := getSSEScannerBuf64K()
	scanner.Buffer(scanBuf[:0], maxLineSize)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			return finalize()
		}
		var chunk apicompat.ChatCompletionsChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			logger.L().Warn("chat completions upstream: failed to parse stream chunk", zap.Error(err))
			continue
		}
		events := apicompat.ChatChunkToResponsesEvents(&chunk, state)
		if state.Usage != nil {
			usage = openAIUsageFromResponsesUsage(state.Usage)
		}
		if err := writeEvents(events); err != nil {
			return &openaiStreamingResult{usage: usage, firstTokenMs: firstTokenMs}, nil
		}
	}
	if err := scanner.Err(); err != nil {
		if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
			logger.L().Warn("chat completions upstream: stream read error", zap.Error(err))
		}
	}
	return finalize()
}

func chatResponsesEventStartsClientOutput(eventType string) bool {
	switch eventType {
	case "response.output_text.delta", "response.function_call_arguments.delta", "response.output_item.added":
		return true
	default:
		return false
	}
}

func openAIUsageFromResponsesUsage(usage *apicompat.ResponsesUsage) *OpenAIUsage {
	if usage == nil {
		return &OpenAIUsage{}
	}
	out := &OpenAIUsage{
		InputTokens:  usage.InputTokens,
		OutputTokens: usage.OutputTokens,
	}
	if usage.InputTokensDetails != nil {
		out.CacheReadInputTokens = usage.InputTokensDetails.CachedTokens
	}
	return out
}
