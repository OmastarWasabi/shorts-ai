package gpt

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"image/png"
	"io"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	imagesURL        = "https://api.openai.com/v1/images/generations"
	maxResponseBytes = 48 << 20
	maxImageBytes    = 32 << 20
)

type apiError struct {
	statusCode int
	code       string
	kind       string
	retryAfter string
}

func (e *apiError) Error() string {
	return fmt.Sprintf("OpenAI Images вернул HTTP %d (%s)", e.statusCode, http.StatusText(e.statusCode))
}

func requestImage(ctx context.Context, client *http.Client, text string, cfg config) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Minute)
	defer cancel()
	for attempt := 0; ; attempt++ {
		data, err := requestImageOnce(ctx, client, text, cfg)
		if err == nil {
			return data, nil
		}
		var apiErr *apiError
		if !errors.As(err, &apiErr) || apiErr.statusCode != http.StatusTooManyRequests || attempt == 2 {
			return nil, err
		}
		if apiErr.code == "insufficient_quota" || apiErr.kind == "insufficient_quota" || strings.HasPrefix(apiErr.code, "billing_") {
			return nil, err
		}
		delay := retryDelay(apiErr.retryAfter, attempt)
		timer := time.NewTimer(delay)
		select {
		case <-ctx.Done():
			timer.Stop()
			return nil, ctx.Err()
		case <-timer.C:
		}
	}
}

func requestImageOnce(ctx context.Context, client *http.Client, text string, cfg config) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	body, err := json.Marshal(struct {
		Model        string `json:"model"`
		Prompt       string `json:"prompt"`
		Size         string `json:"size"`
		Quality      string `json:"quality"`
		N            int    `json:"n"`
		OutputFormat string `json:"output_format"`
	}{cfg.model, text, cfg.size, cfg.quality, 1, "png"})
	if err != nil {
		return nil, fmt.Errorf("подготовка запроса OpenAI: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, imagesURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("создание запроса OpenAI: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+cfg.apiKey)
	req.Header.Set("Content-Type", "application/json")
	res, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("запрос OpenAI Images: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		var payload struct {
			Error struct {
				Code string `json:"code"`
				Type string `json:"type"`
			} `json:"error"`
		}
		_ = json.NewDecoder(io.LimitReader(res.Body, 64<<10)).Decode(&payload)
		return nil, &apiError{res.StatusCode, payload.Error.Code, payload.Error.Type, res.Header.Get("Retry-After")}
	}
	response, err := io.ReadAll(io.LimitReader(res.Body, maxResponseBytes+1))
	if err != nil {
		return nil, fmt.Errorf("получение изображения: %w", err)
	}
	if len(response) > maxResponseBytes {
		return nil, errors.New("ответ OpenAI Images превышает 48 МиБ")
	}
	var payload struct {
		Data []struct {
			Base64 string `json:"b64_json"`
		} `json:"data"`
	}
	if err := json.Unmarshal(response, &payload); err != nil {
		return nil, errors.New("OpenAI Images вернул некорректный JSON")
	}
	if len(payload.Data) != 1 || payload.Data[0].Base64 == "" {
		return nil, errors.New("OpenAI Images не вернул одно изображение в b64_json")
	}
	data, err := base64.StdEncoding.DecodeString(payload.Data[0].Base64)
	if err != nil {
		return nil, errors.New("OpenAI Images вернул некорректный Base64")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return data, nil
}

func retryDelay(header string, attempt int) time.Duration {
	delay := time.Second*time.Duration(1<<attempt) + time.Duration(rand.Intn(250))*time.Millisecond
	if seconds, err := strconv.ParseUint(header, 10, 64); err == nil {
		if seconds > uint64((1<<63-1)/int64(time.Second)) {
			return time.Duration(1<<63 - 1)
		}
		return max(delay, time.Duration(seconds)*time.Second)
	}
	if date, err := http.ParseTime(header); err == nil {
		return max(delay, time.Until(date))
	}
	return delay
}

func validatePNG(data []byte) error {
	if len(data) == 0 || len(data) > maxImageBytes {
		return errors.New("PNG пуст или превышает 32 МиБ")
	}
	cfg, err := png.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return errors.New("некорректный заголовок PNG")
	}
	if cfg.Width > 4096 || cfg.Height > 4096 {
		return errors.New("PNG превышает допустимые размеры 4096×4096")
	}
	if _, err := png.Decode(bytes.NewReader(data)); err != nil {
		return errors.New("повреждённый PNG")
	}
	return nil
}

func imageExists(path string) (bool, error) {
	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return false, nil
	}
	if err != nil {
		return false, fmt.Errorf("проверка %s: %w", path, err)
	}
	if !info.Mode().IsRegular() || info.Size() > maxImageBytes {
		return false, fmt.Errorf("%s должен быть обычным PNG-файлом размером до 32 МиБ", path)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return false, fmt.Errorf("чтение %s: %w", path, err)
	}
	if err := validatePNG(data); err != nil {
		return false, fmt.Errorf("%s: %w; проверьте или удалите файл вручную", path, err)
	}
	return true, nil
}

func saveImage(path string, data []byte) error {
	if err := validatePNG(data); err != nil {
		return err
	}
	file, err := os.CreateTemp(filepath.Dir(path), ".image-*.png")
	if err != nil {
		return fmt.Errorf("создание временного PNG: %w", err)
	}
	defer os.Remove(file.Name())
	_, writeErr := file.Write(data)
	closeErr := file.Close()
	if err := errors.Join(writeErr, closeErr); err != nil {
		return fmt.Errorf("сохранение PNG: %w", err)
	}
	if err := os.Rename(file.Name(), path); err != nil {
		return fmt.Errorf("сохранение %s: %w", path, err)
	}
	return nil
}
