package gpt

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"unicode/utf8"
)

type Prompt struct {
	Number int
	Text   string
}

var promptMarker = regexp.MustCompile(`^\[([0-9]{2,})\]$`)

func ReadPrompts(path string) ([]Prompt, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("чтение %s: %w", path, err)
	}
	if !utf8.Valid(content) {
		return nil, errors.New("prompts.txt должен быть в кодировке UTF-8")
	}
	var prompts []Prompt
	var lines []string
	number := 0
	for lineIndex, line := range strings.Split(strings.TrimPrefix(string(content), "\uFEFF"), "\n") {
		line = strings.TrimSuffix(line, "\r")
		trimmed := strings.TrimSpace(line)
		if marker := promptMarker.FindStringSubmatch(trimmed); marker != nil {
			if number > 0 {
				text := strings.TrimSpace(strings.Join(lines, "\n"))
				if text == "" {
					return nil, fmt.Errorf("промпт [%02d] пуст", number)
				}
				prompts = append(prompts, Prompt{Number: number, Text: text})
				lines = nil
			}
			next, err := strconv.Atoi(marker[1])
			if err != nil || next != number+1 {
				return nil, fmt.Errorf("строка %d: ожидался маркер [%02d]; проверьте порядок и дубликаты", lineIndex+1, number+1)
			}
			number = next
			continue
		}
		if strings.HasPrefix(trimmed, "[") && len(trimmed) > 1 && trimmed[1] >= '0' && trimmed[1] <= '9' {
			return nil, fmt.Errorf("неверный маркер в строке %d; используйте [01] на отдельной строке", lineIndex+1)
		}
		if number == 0 {
			if trimmed != "" {
				return nil, errors.New("prompts.txt должен начинаться с маркера [01]")
			}
			continue
		}
		lines = append(lines, line)
	}
	if number == 0 {
		return nil, errors.New("prompts.txt не содержит промптов")
	}
	text := strings.TrimSpace(strings.Join(lines, "\n"))
	if text == "" {
		return nil, fmt.Errorf("промпт [%02d] пуст", number)
	}
	prompts = append(prompts, Prompt{Number: number, Text: text})
	for _, prompt := range prompts {
		if utf8.RuneCountInString(prompt.Text) > 32000 {
			return nil, fmt.Errorf("промпт [%02d] превышает 32000 символов", prompt.Number)
		}
	}
	return prompts, nil
}
