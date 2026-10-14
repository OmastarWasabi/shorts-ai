package gpt

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

func enterProject(t *testing.T, count int) {
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
	t.Setenv("OPENAI_API_KEY", "test-key")
	t.Setenv("OPENAI_IMAGE_MODEL", "")
	t.Setenv("OPENAI_IMAGE_SIZE", "")
	t.Setenv("OPENAI_IMAGE_QUALITY", "")
	var prompts strings.Builder
	for number := 1; number <= count; number++ {
		fmt.Fprintf(&prompts, "[%02d]\n%d\n\n", number, number)
	}
	if err := os.WriteFile("prompts.txt", []byte(prompts.String()), 0600); err != nil {
		t.Fatal(err)
	}
}

func testPNG(number int) []byte {
	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.Set(0, 0, color.RGBA{R: uint8(number), A: 255})
	var buffer bytes.Buffer
	if err := png.Encode(&buffer, img); err != nil {
		panic(err)
	}
	return buffer.Bytes()
}

func imageResponse(data []byte) *http.Response {
	body := `{"data":[{"b64_json":"` + base64.StdEncoding.EncodeToString(data) + `"}]}`
	return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(body))}
}

func TestGenerateConcurrencyAndResume(t *testing.T) {
	enterProject(t, 7)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	started := make(chan int, 10)
	release := make(chan struct{})
	var mu sync.Mutex
	active, peak, requests := 0, 0, 0
	client := &http.Client{Transport: roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if req.Method != http.MethodPost || req.URL.String() != imagesURL || req.Header.Get("Authorization") != "Bearer test-key" {
			return nil, errors.New("unexpected endpoint or authorization")
		}
		var body struct {
			Model        string `json:"model"`
			Prompt       string `json:"prompt"`
			Size         string `json:"size"`
			Quality      string `json:"quality"`
			N            int    `json:"n"`
			OutputFormat string `json:"output_format"`
		}
		if err := json.NewDecoder(req.Body).Decode(&body); err != nil {
			return nil, err
		}
		if body.Model != "gpt-image-1-mini" || body.Size != "1024x1536" || body.Quality != "medium" || body.N != 1 || body.OutputFormat != "png" {
			return nil, errors.New("unexpected image parameters")
		}
		number, err := strconv.Atoi(body.Prompt)
		if err != nil {
			return nil, err
		}
		mu.Lock()
		active++
		requests++
		peak = max(peak, active)
		mu.Unlock()
		defer func() { mu.Lock(); active--; mu.Unlock() }()
		started <- number
		select {
		case <-release:
			return imageResponse(testPNG(number)), nil
		case <-req.Context().Done():
			return nil, req.Context().Err()
		}
	})}
	finished := make(chan error, 1)
	done := make(chan struct{})
	go func() {
		defer close(done)
		finished <- (Generator{}).Generate(ctx, client)
	}()
	defer func() { cancel(); <-done }()
	for i := 0; i < 3; i++ {
		select {
		case <-started:
		case <-time.After(3 * time.Second):
			t.Fatal("three requests must start concurrently")
		}
	}
	close(release)
	select {
	case err := <-finished:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("generation did not finish")
	}
	if peak != 3 || requests != 7 {
		t.Fatalf("unexpected concurrency or request count: peak=%d, requests=%d", peak, requests)
	}
	for number := 1; number <= 7; number++ {
		data, err := os.ReadFile(imagePath(number))
		if err != nil || !bytes.Equal(data, testPNG(number)) {
			t.Fatalf("image %d saved incorrectly: %v", number, err)
		}
	}
	if err := (Generator{}).Generate(ctx, client); err != nil || requests != 7 {
		t.Fatalf("cached images were regenerated: requests=%d, error=%v", requests, err)
	}
	if err := os.Remove(imagePath(4)); err != nil {
		t.Fatal(err)
	}
	if err := (Generator{}).Generate(ctx, client); err != nil || requests != 8 {
		t.Fatalf("only the missing image must regenerate: requests=%d, error=%v", requests, err)
	}
	if err := os.WriteFile(imagePath(1), []byte("broken PNG"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := (Generator{}).Generate(ctx, client); err == nil || requests != 8 {
		t.Fatal("corrupt existing PNG must stop before new requests")
	}
	if temporary, _ := filepath.Glob("images/.image-*.png"); len(temporary) != 0 {
		t.Fatalf("temporary files left behind: %v", temporary)
	}
}

func TestGenerateRejectsBeforeRequests(t *testing.T) {
	for _, name := range []string{"invalid prompts", "missing key", "invalid size", "invalid quality"} {
		t.Run(name, func(t *testing.T) {
			enterProject(t, 1)
			switch name {
			case "invalid prompts":
				if err := os.WriteFile("prompts.txt", []byte("[01]\nFirst.\n[01]\nDuplicate."), 0600); err != nil {
					t.Fatal(err)
				}
			case "missing key":
				t.Setenv("OPENAI_API_KEY", "")
			case "invalid size":
				t.Setenv("OPENAI_IMAGE_SIZE", "wrong")
			case "invalid quality":
				t.Setenv("OPENAI_IMAGE_QUALITY", "wrong")
			}
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				t.Error("invalid input must not send requests")
				return nil, errors.New("unexpected request")
			})}
			if err := (Generator{}).Generate(context.Background(), client); err == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}

func TestGenerateFailuresDoNotSaveImages(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{"unauthorized", 401, `{"error":{"code":"invalid_api_key"}}`},
		{"quota", 429, `{"error":{"code":"insufficient_quota"}}`},
		{"server error", 503, `{}`},
		{"bad JSON", 200, `not JSON`},
		{"missing image", 200, `{"data":[]}`},
		{"bad Base64", 200, `{"data":[{"b64_json":"!"}]}`},
		{"bad PNG", 200, `{"data":[{"b64_json":"bm90IGEgcG5n"}]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			enterProject(t, 1)
			requests := 0
			client := &http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
				requests++
				return &http.Response{StatusCode: tc.status, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
			})}
			if err := (Generator{}).Generate(context.Background(), client); err == nil {
				t.Fatal("expected generation error")
			}
			if requests != 1 {
				t.Fatalf("this failure must not retry: %d requests", requests)
			}
			if _, err := os.Stat(imagePath(1)); !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("failed image must not be saved: %v", err)
			}
		})
	}
}
