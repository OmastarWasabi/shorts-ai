package gpt

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestRateLimitRetries(t *testing.T) {
	for _, succeeds := range []bool{true, false} {
		name := "exhausted"
		if succeeds {
			name = "recovers"
		}
		t.Run(name, func(t *testing.T) {
			requests := 0
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				requests++
				if succeeds && requests == 2 {
					return imageResponse(testPNG(1)), nil
				}
				return &http.Response{
					StatusCode: 429,
					Header:     http.Header{"Retry-After": {"0"}},
					Body:       io.NopCloser(strings.NewReader(`{"error":{"code":"rate_limit_exceeded"}}`)),
				}, nil
			})}
			_, err := requestImage(context.Background(), client, "Prompt.", config{apiKey: "test-key"})
			if succeeds && (err != nil || requests != 2) {
				t.Fatalf("expected successful retry: requests=%d, error=%v", requests, err)
			}
			if !succeeds && (err == nil || requests != 3) {
				t.Fatalf("expected at most three attempts: requests=%d, error=%v", requests, err)
			}
		})
	}
}

func TestCancellationInterruptsRetryDelay(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	requests := 0
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		requests++
		cancel()
		return &http.Response{StatusCode: 429, Header: http.Header{"Retry-After": {"60"}}, Body: io.NopCloser(strings.NewReader(`{}`))}, nil
	})}
	_, err := requestImage(ctx, client, "Prompt.", config{})
	if !errors.Is(err, context.Canceled) || requests != 1 {
		t.Fatalf("cancellation must stop retries: requests=%d, error=%v", requests, err)
	}
}

func TestRetryDelay(t *testing.T) {
	if delay := retryDelay("10", 0); delay != 10*time.Second {
		t.Fatalf("Retry-After seconds not honored: %v", delay)
	}
	header := time.Now().Add(10 * time.Second).UTC().Format(http.TimeFormat)
	if delay := retryDelay(header, 0); delay < 8*time.Second || delay > 10*time.Second {
		t.Fatalf("Retry-After date not honored: %v", delay)
	}
	if delay := retryDelay("invalid", 1); delay < 2*time.Second || delay >= 2250*time.Millisecond {
		t.Fatalf("unexpected fallback delay: %v", delay)
	}
}

func TestValidatePNGRejectsTruncation(t *testing.T) {
	data := testPNG(1)
	if err := validatePNG(data[:len(data)-12]); err == nil {
		t.Fatal("truncated PNG must not be accepted")
	}
}
