package fish

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"unicode/utf8"
)

const (
	ttsURL        = "https://api.fish.audio/v1/tts"
	voicePath     = "voice.mp3"
	maxAudioBytes = 32 << 20
)

type Generator struct{}

type config struct {
	apiKey      string
	referenceID string
	model       string
}

func (Generator) Generate(ctx context.Context, client *http.Client) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if client == nil {
		return errors.New("HTTP-клиент не задан")
	}
	text, err := ReadText("story.txt")
	if err != nil {
		return err
	}
	if info, err := os.Stat(voicePath); err == nil {
		if !info.Mode().IsRegular() || info.Size() == 0 {
			return errors.New("voice.mp3 должен быть непустым обычным файлом; проверьте его вручную")
		}
		fmt.Fprintln(os.Stderr, "voice.mp3 уже существует — генерация пропущена. Для озвучки нового текста удалите его вручную.")
		return nil
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("проверка voice.mp3: %w", err)
	}

	cfg, err := readConfig()
	if err != nil {
		return err
	}
	audio, err := requestAudio(ctx, client, text, cfg)
	if err != nil {
		return err
	}
	return saveAudio(audio)
}

func ReadText(path string) (string, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("чтение %s: %w", path, err)
	}
	text := strings.TrimPrefix(string(content), "\uFEFF")
	if strings.TrimSpace(text) == "" {
		return "", errors.New("текст для озвучки не должен быть пустым")
	}
	if !utf8.ValidString(text) {
		return "", errors.New("текст для озвучки должен быть в кодировке UTF-8")
	}
	return text, nil
}

func readConfig() (config, error) {
	cfg := config{
		apiKey:      os.Getenv("FISH_API_KEY"),
		referenceID: os.Getenv("FISH_REFERENCE_ID"),
		model:       os.Getenv("FISH_MODEL"),
	}
	if strings.TrimSpace(cfg.apiKey) == "" {
		return config{}, errors.New("задайте FISH_API_KEY в окружении")
	}
	if cfg.referenceID == "" {
		cfg.referenceID = "8336207c3431432cb60e8851e78ba437"
	}
	if cfg.model == "" {
		cfg.model = "s2.1-pro-free"
	}
	switch cfg.model {
	case "s1", "s2-pro", "s2.1-pro", "s2.1-pro-free", "drama-3-preview":
	default:
		return config{}, errors.New("неподдерживаемый FISH_MODEL; проверьте название модели в документации Fish Audio")
	}
	return cfg, nil
}

func requestAudio(ctx context.Context, client *http.Client, text string, cfg config) ([]byte, error) {
	body, err := json.Marshal(struct {
		Text        string `json:"text"`
		ReferenceID string `json:"reference_id"`
		Format      string `json:"format"`
	}{Text: text, ReferenceID: cfg.referenceID, Format: "mp3"})
	if err != nil {
		return nil, fmt.Errorf("подготовка запроса Fish Audio: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, ttsURL, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("создание запроса Fish Audio: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+cfg.apiKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("model", cfg.model)

	res, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("запрос Fish Audio: %w", err)
	}
	defer res.Body.Close()
	if res.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Fish Audio вернул HTTP %d (%s)", res.StatusCode, http.StatusText(res.StatusCode))
	}
	contentType := strings.ToLower(res.Header.Get("Content-Type"))
	if strings.Contains(contentType, "json") || strings.Contains(contentType, "text/") {
		return nil, errors.New("Fish Audio вернул текст вместо MP3")
	}
	audio, err := io.ReadAll(io.LimitReader(res.Body, maxAudioBytes+1))
	if err != nil {
		return nil, fmt.Errorf("получение MP3: %w", err)
	}
	if len(audio) == 0 || len(audio) > maxAudioBytes {
		return nil, errors.New("Fish Audio вернул пустое или слишком большое аудио")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return audio, nil
}

func saveAudio(audio []byte) error {
	file, err := os.CreateTemp(".", ".voice-*.mp3")
	if err != nil {
		return fmt.Errorf("создание временного MP3: %w", err)
	}
	defer os.Remove(file.Name())
	_, writeErr := file.Write(audio)
	closeErr := file.Close()
	if err := errors.Join(writeErr, closeErr); err != nil {
		return fmt.Errorf("сохранение MP3: %w", err)
	}
	if err := os.Rename(file.Name(), voicePath); err != nil {
		return fmt.Errorf("сохранение voice.mp3: %w", err)
	}
	return nil
}
