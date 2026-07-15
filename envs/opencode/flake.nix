{
  description = "OpenCode wired to the OSU LiteLLM proxy (OpenAI-compatible), default model Opus 4.8";

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
          packages = [ pkgs.opencode ];

          shellHook = ''
            # Load LITELLM_URL / LITELLM_KEY from the repository-root .env.
            root="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
            if [ -f "$root/.env" ]; then
              set -a; . "$root/.env"; set +a
            fi

            if [ -z "''${LITELLM_KEY:-}" ]; then
              echo "warning: LITELLM_KEY not set (expected in $root/.env)" >&2
            fi

            # Point OpenCode at the project-local config that defines the
            # `litellm` custom provider. It reads LITELLM_KEY via {env:LITELLM_KEY}.
            export OPENCODE_CONFIG="$root/envs/opencode/opencode.json"

            echo "opencode $(opencode --version 2>/dev/null || echo '?')  ->  litellm/claude-opus-4-8  (config: $OPENCODE_CONFIG)"
          '';
        };
      });
}
