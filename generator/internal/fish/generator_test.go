package fish

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func enterProject(t *testing.T) {
	t.Helper()
	previous, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(previous); err != nil {
			t.Error(err)
		}
	})
	t.Setenv("FISH_API_KEY", "test-key")
	t.Setenv("FISH_REFERENCE_ID", "test-voice")
	t.Setenv("FISH_MODEL", "")
	if err := os.WriteFile("story.txt", []byte("Текст для озвучки."), 0600); err != nil {
		t.Fatal(err)
	}
}

func TestGenerateAndReuse(t *testing.T) {
	enterProject(t)
	audio := []byte("ID3-test-audio")
	requests := 0
	text := "Привет! Это обычный текст\nбез маркеров сцен."
	if err := os.WriteFile("story.txt", []byte("\uFEFF"+text), 0600); err != nil {
		t.Fatal(err)
	}
	g := Generator{}
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		requests++
		if req.Method != http.MethodPost || req.URL.String() != ttsURL {
			t.Fatalf("unexpected endpoint: %s %s", req.Method, req.URL)
		}
		if req.Header.Get("Authorization") != "Bearer test-key" || req.Header.Get("model") != "s2.1-pro-free" || req.Header.Get("Content-Type") != "application/json" {
			t.Fatal("unexpected request headers")
		}
		var body struct {
			Text        string `json:"text"`
			ReferenceID string `json:"reference_id"`
			Format      string `json:"format"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			t.Fatal(err)
		}
		if body.Text != text || body.ReferenceID != "test-voice" || body.Format != "mp3" {
			t.Fatalf("unexpected request body: %+v", body)
		}
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{"Content-Type": {"audio/mpeg"}}, Body: io.NopCloser(bytes.NewReader(audio))}, nil
	})}
	for i := 0; i < 2; i++ {
		if err := g.Generate(context.Background(), client); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile("story.txt", []byte("Другой текст."), 0600); err != nil {
		t.Fatal(err)
	}
	if err := g.Generate(context.Background(), client); err != nil {
		t.Fatal(err)
	}
	saved, err := os.ReadFile(voicePath)
	if err != nil || !bytes.Equal(saved, audio) || requests != 1 {
		t.Fatalf("audio changed or regenerated: requests=%d, error=%v", requests, err)
	}
	if temporary, _ := filepath.Glob(".voice-*.mp3"); len(temporary) != 0 {
		t.Fatalf("temporary files left behind: %v", temporary)
	}
}

func TestGenerateFailuresDoNotSaveAudio(t *testing.T) {
	cases := []struct {
		name        string
		status      int
		contentType string
		body        string
	}{
		{"unauthorized", 401, "application/json", `{"message":"unauthorized"}`},
		{"no credits", 402, "application/json", `{"message":"no credits"}`},
		{"rate limited", 429, "application/json", `{"message":"rate limited"}`},
		{"server error", 503, "application/json", `{"message":"unavailable"}`},
		{"empty audio", 200, "audio/mpeg", ""},
		{"JSON instead of audio", 200, "application/json", `{"message":"error"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enterProject(t)
			requests := 0
			g := Generator{}
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				requests++
				return &http.Response{StatusCode: tc.status, Header: http.Header{"Content-Type": {tc.contentType}}, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
			})}
			if err := g.Generate(context.Background(), client); err == nil {
				t.Fatal("expected generation error")
			}
			if _, err := os.Stat(voicePath); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("failed request must not leave voice.mp3: %v", err)
			}
			if requests != 1 {
				t.Fatalf("failed requests must not retry automatically: %d", requests)
			}
		})
	}
}

type interruptedAudio struct{}

func (interruptedAudio) Read([]byte) (int, error) {
	return 0, io.ErrUnexpectedEOF
}

func TestInterruptedResponseDoesNotSaveAudio(t *testing.T) {
	enterProject(t)
	g := Generator{}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		body := io.MultiReader(strings.NewReader("ID3-partial"), interruptedAudio{})
		return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(body)}, nil
	})}
	if err := g.Generate(context.Background(), client); !errors.Is(err, io.ErrUnexpectedEOF) {
		t.Fatalf("expected interrupted response error: %v", err)
	}
	if _, err := os.Stat(voicePath); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("interrupted response must not leave voice.mp3: %v", err)
	}
}

func TestInvalidConfigurationDoesNotRequest(t *testing.T) {
	enterProject(t)
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("invalid configuration must not send a request")
		return nil, errors.New("unexpected request")
	})}
	for _, tc := range []struct{ name, key, model, text string }{
		{"missing key", "", "", "Текст."},
		{"unknown model", "test-key", "typo", "Текст."},
		{"empty text", "test-key", "", " \n\t"},
		{"invalid UTF-8", "test-key", "", "\xff"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv("FISH_API_KEY", tc.key)
			t.Setenv("FISH_MODEL", tc.model)
			if err := os.WriteFile("story.txt", []byte(tc.text), 0600); err != nil {
				t.Fatal(err)
			}
			if err := (Generator{}).Generate(context.Background(), client); err == nil {
				t.Fatal("expected configuration error")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	g := Generator{}
	if err := g.Generate(ctx, client); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context cancellation: %v", err)
	}
	if err := g.Generate(context.Background(), nil); err == nil {
		t.Fatal("nil client must produce an error")
	}
}

func TestMissingScriptDoesNotRequest(t *testing.T) {
	enterProject(t)
	if err := os.Remove("story.txt"); err != nil {
		t.Fatal(err)
	}
	client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		t.Fatal("missing script must not send a request")
		return nil, errors.New("unexpected request")
	})}
	if err := (Generator{}).Generate(context.Background(), client); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("expected missing script error: %v", err)
	}
}
