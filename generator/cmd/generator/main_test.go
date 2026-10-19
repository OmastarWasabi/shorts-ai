package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"fmt"
	"image"
	"image/png"
	"io"
	"net/http"
	"os"
	"sync/atomic"
	"testing"
)

func TestValidateInputs(t *testing.T) {
	cfg := config{GenerateVoice: true, GenerateImages: true}
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
	if err := validateInputs(cfg); err == nil {
		t.Fatal("missing input files must fail")
	}
	if err := os.WriteFile("story.txt", []byte("Обычный текст."), 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile("prompts.txt", []byte("[01]\nA scene.\n[01]\nDuplicate."), 0600); err != nil {
		t.Fatal(err)
	}
	if err := validateInputs(cfg); err == nil {
		t.Fatal("invalid prompts must fail before either generator starts")
	}
	if err := os.WriteFile("prompts.txt", []byte("[01]\nA scene."), 0600); err != nil {
		t.Fatal(err)
	}
	if err := validateInputs(cfg); err != nil {
		t.Fatal(err)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func TestRunGenerationModes(t *testing.T) {
	var pngData bytes.Buffer
	if err := png.Encode(&pngData, image.NewRGBA(image.Rect(0, 0, 2, 2))); err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name           string
		cfg            config
		missingPrompts bool
	}{
		{name: "voice only", cfg: config{GenerateVoice: true}},
		{name: "images only", cfg: config{GenerateImages: true}},
		{name: "both", cfg: config{GenerateVoice: true, GenerateImages: true}},
		{name: "neither"},
		{name: "invalid inputs prevent all requests", cfg: config{GenerateVoice: true, GenerateImages: true}, missingPrompts: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
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
			// Отключённые генераторы не должны читать отсутствующие файлы и ключи.
			t.Setenv("FISH_API_KEY", "")
			t.Setenv("OPENAI_API_KEY", "")
			if tt.cfg.GenerateVoice {
				t.Setenv("FISH_API_KEY", "test-fish")
				t.Setenv("FISH_MODEL", "s2.1-pro-free")
				if err := os.WriteFile("story.txt", []byte("Тест."), 0600); err != nil {
					t.Fatal(err)
				}
			}
			if tt.cfg.GenerateImages && !tt.missingPrompts {
				t.Setenv("OPENAI_API_KEY", "test-openai")
				t.Setenv("OPENAI_IMAGE_SIZE", "1024x1536")
				t.Setenv("OPENAI_IMAGE_QUALITY", "low")
				if err := os.WriteFile("prompts.txt", []byte("[01]\nA scene."), 0600); err != nil {
					t.Fatal(err)
				}
			}
			var voiceCalls, imageCalls atomic.Int32
			client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
				var body, contentType string
				switch req.URL.Host {
				case "api.fish.audio":
					voiceCalls.Add(1)
					body, contentType = "ID3test-audio", "audio/mpeg"
				case "api.openai.com":
					imageCalls.Add(1)
					body = fmt.Sprintf(`{"data":[{"b64_json":%q}]}`, base64.StdEncoding.EncodeToString(pngData.Bytes()))
					contentType = "application/json"
				default:
					return nil, fmt.Errorf("unexpected host %s", req.URL.Host)
				}
				return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(bytes.NewBufferString(body))}, nil
			})}
			err = runGeneration(context.Background(), client, tt.cfg)
			if (err != nil) != tt.missingPrompts {
				t.Fatalf("runGeneration error = %v", err)
			}
			wantVoice, wantImages := int32(0), int32(0)
			if !tt.missingPrompts {
				if tt.cfg.GenerateVoice {
					wantVoice = 1
				}
				if tt.cfg.GenerateImages {
					wantImages = 1
				}
			}
			if voiceCalls.Load() != wantVoice || imageCalls.Load() != wantImages {
				t.Fatalf("requests: voice=%d images=%d, want %d/%d", voiceCalls.Load(), imageCalls.Load(), wantVoice, wantImages)
			}
			for path, wanted := range map[string]bool{"voice.mp3": wantVoice > 0, "images/01.png": wantImages > 0} {
				_, err := os.Stat(path)
				if (err == nil) != wanted {
					t.Errorf("output %s: exists=%v, want %v", path, err == nil, wanted)
				}
			}
		})
	}
}
