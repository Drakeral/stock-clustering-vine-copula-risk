# Credential handling

The downloader reads credentials only from environment variables:

| Variable | Purpose |
|---|---|
| `MASSIVE_API_KEY` | REST reference endpoints |
| `MASSIVE_S3_ACCESS_KEY` | Flat-file S3 access-key identifier |
| `MASSIVE_S3_SECRET_KEY` | Flat-file S3 secret |

No credential value, authenticated URL, home-directory path, or environment
dump is written to a manifest or audit file.

## Portable local file setup

`scripts/load_credentials.zsh` uses the XDG configuration directory when set,
otherwise `${HOME}/.config`. Store the files below
`${XDG_CONFIG_HOME:-${HOME}/.config}/fe5110/`:

- `massive_api_key.txt`: one line containing the REST key;
- `massive_flat_file_key.txt`: first line access-key ID, second line secret key.

For example, create the directory with owner-only permissions and then edit the
two files using a local password-aware editor:

```zsh
FE5110_CONFIG_BASE="${XDG_CONFIG_HOME:-${HOME}/.config}"
umask 077
mkdir -p "${FE5110_CONFIG_BASE}/fe5110"
chmod 700 "${FE5110_CONFIG_BASE}/fe5110"
touch "${FE5110_CONFIG_BASE}/fe5110/massive_api_key.txt"
touch "${FE5110_CONFIG_BASE}/fe5110/massive_flat_file_key.txt"
chmod 600 "${FE5110_CONFIG_BASE}/fe5110/massive_api_key.txt"
chmod 600 "${FE5110_CONFIG_BASE}/fe5110/massive_flat_file_key.txt"
unset FE5110_CONFIG_BASE
```

Load them into the current interactive zsh session:

```zsh
source scripts/load_credentials.zsh
```

The loader fails if either file is unreadable. It does not print values. The
variables live only in the current process and its children; open a fresh shell
or `unset MASSIVE_API_KEY MASSIVE_S3_ACCESS_KEY MASSIVE_S3_SECRET_KEY` after use.

## Alternative secret managers

CI or another shell may inject the same three environment variables from its
native secret store and skip the loader. `.env.example` documents names only;
the project does not automatically parse `.env` files.

Never put values in source code, notebooks, command-line arguments, screenshots,
configuration, shell history, Git, the submission bundle, or issue text. The
ignore rules cover common local secret filenames as a secondary safeguard, not
as a substitute for keeping secrets outside the repository.
