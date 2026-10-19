package main

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

type config struct {
	GenerateVoice  bool
	GenerateImages bool
}

func readConfig() (config, error) {
	voice, err := readBoolEnv("GENERATE_VOICE")
	if err != nil {
		return config{}, err
	}
	images, err := readBoolEnv("GENERATE_IMAGES")
	if err != nil {
		return config{}, err
	}
	return config{GenerateVoice: voice, GenerateImages: images}, nil
}

func readBoolEnv(name string) (bool, error) {
	value, exists := os.LookupEnv(name)
	if !exists {
		return true, nil
	}
	enabled, err := strconv.ParseBool(strings.TrimSpace(value))
	if err != nil {
		return false, fmt.Errorf("%s должен быть true или false", name)
	}
	return enabled, nil
}
