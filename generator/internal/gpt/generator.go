package gpt

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

type Generator struct{}

type config struct {
	apiKey  string
	model   string
	size    string
	quality string
}

func (Generator) Generate(ctx context.Context, client *http.Client) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if client == nil {
		return errors.New("HTTP-клиент не задан")
	}
	prompts, err := ReadPrompts("prompts.txt")
	if err != nil {
		return err
	}
	var pending []Prompt
	for _, prompt := range prompts {
		exists, err := imageExists(imagePath(prompt.Number))
		if err != nil {
			return err
		}
		if !exists {
			pending = append(pending, prompt)
		}
	}
	if len(pending) == 0 {
		return nil
	}
	cfg, err := readConfig()
	if err != nil {
		return err
	}
	if err := os.MkdirAll("images", 0755); err != nil {
		return fmt.Errorf("создание images: %w", err)
	}
	fmt.Fprintf(os.Stderr, "OpenAI Images: платная генерация %d изображений, максимум 3 запроса одновременно.\n", len(pending))

	slots := make(chan struct{}, 3)
	errCh := make(chan error, len(pending))
	var wg sync.WaitGroup
	var generationErrors []error
schedule:
	for _, prompt := range pending {
		select {
		case slots <- struct{}{}:
		case <-ctx.Done():
			generationErrors = append(generationErrors, ctx.Err())
			break schedule
		}
		wg.Add(1)
		go func(prompt Prompt) {
			defer wg.Done()
			defer func() { <-slots }()
			data, err := requestImage(ctx, client, prompt.Text, cfg)
			if err == nil {
				err = saveImage(imagePath(prompt.Number), data)
			}
			if err != nil {
				errCh <- fmt.Errorf("изображение %02d: %w", prompt.Number, err)
			}
		}(prompt)
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		generationErrors = append(generationErrors, err)
	}
	return errors.Join(generationErrors...)
}

func imagePath(number int) string {
	return filepath.Join("images", fmt.Sprintf("%02d.png", number))
}

func readConfig() (config, error) {
	cfg := config{
		apiKey:  os.Getenv("OPENAI_API_KEY"),
		model:   os.Getenv("OPENAI_IMAGE_MODEL"),
		size:    os.Getenv("OPENAI_IMAGE_SIZE"),
		quality: os.Getenv("OPENAI_IMAGE_QUALITY"),
	}
	if strings.TrimSpace(cfg.apiKey) == "" {
		return config{}, errors.New("задайте OPENAI_API_KEY в окружении или .env")
	}
	if cfg.model == "" {
		cfg.model = "gpt-image-2.5-sunburst"
	}
	if cfg.size == "" {
		cfg.size = "1024x1536"
	}
	if cfg.quality == "" {
		cfg.quality = "low"
	}
	switch cfg.size {
	case "1024x1024", "1024x1536", "1536x1024":
	default:
		return config{}, errors.New("OPENAI_IMAGE_SIZE должен быть 1024x1024, 1024x1536 или 1536x1024")
	}
	switch cfg.quality {
	case "low", "medium", "high":
	default:
		return config{}, errors.New("OPENAI_IMAGE_QUALITY должен быть low, medium или high")
	}
	return cfg, nil
}
