package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"time"

	"history-glitch/internal/fish"
	"history-glitch/internal/gpt"
)

type FileGenerator interface {
	Generate(ctx context.Context, client *http.Client) error
}

func main() {
	if len(os.Args) != 1 {
		fmt.Fprintln(os.Stderr, "Запускайте программу без аргументов из папки generator; генерации выбираются в .env")
		os.Exit(1)
	}
	if err := loadEnv(".env"); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
	cfg, err := readConfig()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	client := &http.Client{
		Timeout: 3 * time.Minute,
		CheckRedirect: func(*http.Request, []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}

	if err := runGeneration(ctx, client, cfg); err != nil {
		fmt.Fprintln(os.Stderr, "Ошибка:", err)
		os.Exit(1)
	}
	switch {
	case cfg.GenerateVoice && cfg.GenerateImages:
		fmt.Println("Готово: voice.mp3 и изображения в images/")
	case cfg.GenerateVoice:
		fmt.Println("Готово: voice.mp3")
	case cfg.GenerateImages:
		fmt.Println("Готово: изображения в images/")
	default:
		fmt.Println("Все генерации отключены; запросы не выполнялись.")
	}
}

func runGeneration(ctx context.Context, client *http.Client, cfg config) error {
	if err := validateInputs(cfg); err != nil {
		return err
	}
	var generators []FileGenerator
	if cfg.GenerateVoice {
		generators = append(generators, fish.Generator{})
	}
	if cfg.GenerateImages {
		generators = append(generators, gpt.Generator{})
	}
	if len(generators) == 0 {
		return nil
	}
	return runGenerators(ctx, client, generators)
}

func validateInputs(cfg config) error {
	var storyErr, promptsErr error
	if cfg.GenerateVoice {
		_, storyErr = fish.ReadText("story.txt")
	}
	if cfg.GenerateImages {
		_, promptsErr = gpt.ReadPrompts("prompts.txt")
	}
	return errors.Join(storyErr, promptsErr)
}

func runGenerators(ctx context.Context, client *http.Client, generators []FileGenerator) error {
	if len(generators) == 0 {
		return errors.New("генераторы GPT и Fish ещё не подключены")
	}
	if err := ctx.Err(); err != nil {
		return err
	}

	errCh := make(chan error, len(generators))
	var wg sync.WaitGroup

	for _, generator := range generators {
		wg.Add(1)
		go func(generator FileGenerator) {
			defer wg.Done()
			if err := generator.Generate(ctx, client); err != nil {
				errCh <- fmt.Errorf("генератор %T: %w", generator, err)
			}
		}(generator)
	}

	wg.Wait()
	close(errCh)

	var generationErrors []error
	for err := range errCh {
		generationErrors = append(generationErrors, err)
	}
	return errors.Join(generationErrors...)
}
