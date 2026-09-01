#!/bin/zsh
# Source this file from zsh before running authenticated data commands:
#   source scripts/load_credentials.zsh

FE5110_CONFIG_BASE="${XDG_CONFIG_HOME:-${HOME}/.config}"
FE5110_CREDENTIAL_DIR="${FE5110_CONFIG_BASE}/fe5110"
FE5110_REST_KEY_FILE="${FE5110_CREDENTIAL_DIR}/massive_api_key.txt"
FE5110_FLAT_KEY_FILE="${FE5110_CREDENTIAL_DIR}/massive_flat_file_key.txt"

if [[ ! -r "${FE5110_REST_KEY_FILE}" || ! -r "${FE5110_FLAT_KEY_FILE}" ]]; then
  print -u2 "FE5110 credentials not found in ${FE5110_CREDENTIAL_DIR}"
  return 1
fi

IFS= read -r MASSIVE_API_KEY < "${FE5110_REST_KEY_FILE}"
{
  IFS= read -r MASSIVE_S3_ACCESS_KEY
  IFS= read -r MASSIVE_S3_SECRET_KEY
} < "${FE5110_FLAT_KEY_FILE}"

export MASSIVE_API_KEY MASSIVE_S3_ACCESS_KEY MASSIVE_S3_SECRET_KEY
unset FE5110_CONFIG_BASE FE5110_CREDENTIAL_DIR FE5110_REST_KEY_FILE FE5110_FLAT_KEY_FILE
