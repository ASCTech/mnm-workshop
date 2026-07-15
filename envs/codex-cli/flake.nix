{
  description = "Codex CLI wired to the OSU LiteLLM proxy (OpenAI-compatible), model GPT-5.5";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
      in {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.codex ];

          shellHook = ''
            root="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
            if [ -f "$root/.env" ]; then
              set -a; . "$root/.env"; set +a
            fi
            if [ -z "''${LITELLM_KEY:-}" ]; then
              echo "warning: LITELLM_KEY not set (expected in $root/.env)" >&2
            fi

            envdir="$root/envs/codex-cli"

            # Keep Codex config/state project-local instead of ~/.codex, and seed
            # config.toml from the tracked template. The key is read via env_key.
            export CODEX_HOME="$envdir/.codex"
            mkdir -p "$CODEX_HOME"
            cp -f "$envdir/config.toml" "$CODEX_HOME/config.toml"

            echo "codex $(codex --version 2>/dev/null || echo '?')  ->  gpt-5.5-2026-04-24 via LiteLLM  (CODEX_HOME: $CODEX_HOME)"
          '';
        };
      });
}
