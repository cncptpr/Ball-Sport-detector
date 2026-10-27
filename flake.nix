{
  description = "Map prototype FHS development environment";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
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
        '';
        runScript = "$SHELL";
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
