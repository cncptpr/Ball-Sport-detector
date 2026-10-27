{
  description = "Map prototype FHS development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      pythonEnv = pkgs.python312;
      fhs = pkgs.buildFHSEnv {
        name = "map-prototype-fhs";
        targetPkgs = pkgs: with pkgs; [
          graphviz
          pythonEnv
          nodejs_24
          uv
          stdenv.cc.cc.lib
          zlib
          libxcb
          libGL
          glib
          libX11
          libXext
          libSM
          libICE
          libXext
        ];
        profile = ''
          export UV_PYTHON="${pythonEnv}/bin/python3"
          # Use the host's loaded NVIDIA driver; the FHS must not replace it
          # with a different driver version than the running kernel module.
          export NVIDIA_DRIVER_LIB="/nix/store/3lzf85a5xyxafkqlyxw7h9f9pns6kl1i-nvidia-x11-595.99.02/lib"
          if [ -d "$PWD/.venv-gpu/lib/python3.12/site-packages/nvidia" ]; then
            export LD_LIBRARY_PATH="$NVIDIA_DRIVER_LIB:$(find "$PWD/.venv-gpu/lib/python3.12/site-packages/nvidia" -type d -name lib -printf '%p:')$LD_LIBRARY_PATH"
          fi
        '';
        runScript = "${pkgs.bash}/bin/bash";
      };
    in
    {
      packages.${system}.fhs = fhs;
      devShells.${system}.default = pkgs.mkShell {
        packages = [ fhs ];
        shellHook = ''
          # echo "Enter the FHS environment with: map-prototype-fhs"
          exec map-prototype-fhs
        '';
      };
    };
}
