{
  description = "Rust development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

  };

  outputs = { fenix, nixpkgs, ... }:
    let
      systems = [
        "aarch64-darwin"
        "aarch64-linux"
        "x86_64-darwin"
        "x86_64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = import nixpkgs { inherit system; };
          rustToolchain = fenix.packages.${system}.latest.withComponents [
            "cargo"
            "clippy"
            "llvm-tools-preview"
            "rust-analyzer"
            "rust-src"
            "rustc"
            "rustfmt"
          ];
        in
        {
          default = pkgs.mkShell {
            packages = [
              rustToolchain
              pkgs.cargo-llvm-cov
              pkgs.just
            ];
            hardeningDisable = [ "fortify" ];
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          };
        });
    };
}
