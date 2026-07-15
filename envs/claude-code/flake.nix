{
  description = "Claude Code wired to the OSU LiteLLM proxy (Anthropic-compatible), default model Opus 4.8";

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
          packages = [ pkgs.claude-code ];

          # Model / endpoint config. The API key is read from the repo-root .env
          # at shell entry (see shellHook) so no secret is baked into the flake.
          ANTHROPIC_MODEL = "claude-opus-4-8";
          ANTHROPIC_SMALL_FAST_MODEL = "claude-haiku-4-5-20251001";

          shellHook = ''
            # Load LITELLM_URL / LITELLM_KEY from the repository-root .env.
            root="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
            if [ -f "$root/.env" ]; then
              set -a; . "$root/.env"; set +a
            fi

            if [ -z "''${LITELLM_KEY:-}" ]; then
              echo "warning: LITELLM_KEY not set (expected in $root/.env)" >&2
            fi

            # Claude Code appends /v1/messages to ANTHROPIC_BASE_URL, so pass the
            # bare host. The OSU LiteLLM proxy exposes the Anthropic Messages API there.
            export ANTHROPIC_BASE_URL="''${LITELLM_URL%/}"
            export ANTHROPIC_AUTH_TOKEN="''${LITELLM_KEY:-}"
            # Avoid a stale x-api-key from the ambient environment overriding the token.
            unset ANTHROPIC_API_KEY

            echo "claude-code $(claude --version 2>/dev/null || echo '?')  ->  $ANTHROPIC_BASE_URL  model=$ANTHROPIC_MODEL"
          '';
        };
      });
}
