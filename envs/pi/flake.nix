{
  description = "Pi coding agent wired to the OSU LiteLLM proxy (OpenAI-compatible), default model Opus 4.8";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        # Pi (@earendil-works/pi-coding-agent) is not in nixpkgs, so we provide
        # Node.js and install the CLI into a project-local npm prefix on first entry.
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.nodejs_22 pkgs.git ];

          shellHook = ''
            root="$(git rev-parse --show-toplevel 2>/dev/null || echo "$PWD")"
            if [ -f "$root/.env" ]; then
              set -a; . "$root/.env"; set +a
            fi
            if [ -z "''${LITELLM_KEY:-}" ]; then
              echo "warning: LITELLM_KEY not set (expected in $root/.env)" >&2
            fi

            envdir="$root/envs/pi"

            # Install the Pi CLI into a project-local prefix (idempotent).
            export NPM_CONFIG_PREFIX="$envdir/.npm"
            export PATH="$NPM_CONFIG_PREFIX/bin:$PATH"
            if [ ! -x "$NPM_CONFIG_PREFIX/bin/pi" ]; then
              echo "installing @earendil-works/pi-coding-agent (first run only)…"
              npm install -g --ignore-scripts @earendil-works/pi-coding-agent >/dev/null 2>&1 \
                || echo "warning: pi install failed (need network on first run)" >&2
            fi

            # Keep Pi's config/state project-local instead of ~/.pi/agent, and
            # seed it from the tracked templates. Pi interpolates $LITELLM_KEY at read.
            export PI_CODING_AGENT_DIR="$envdir/.pi-agent"
            mkdir -p "$PI_CODING_AGENT_DIR"
            cp -f "$envdir/models.json"   "$PI_CODING_AGENT_DIR/models.json"
            cp -f "$envdir/settings.json" "$PI_CODING_AGENT_DIR/settings.json"

            echo "pi $("$NPM_CONFIG_PREFIX/bin/pi" --version 2>/dev/null || echo '?')  ->  litellm/claude-opus-4-8  (dir: $PI_CODING_AGENT_DIR)"
          '';
        };
      });
}
