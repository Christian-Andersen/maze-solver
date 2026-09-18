set dotenv-load

alias c := check
alias r := run

[private]
default:
    @just --list

setup:
    uv python install 3.14

check:
    uv run -- prek run --all-files

run *args:
    uv run -- main.py {{ args }}

audit:
    uv run -- prek run audit

ci: setup
    uv run -- prek run --all-files --stage manual
