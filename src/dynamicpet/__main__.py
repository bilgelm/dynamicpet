"""Command-line interface."""
import click


@click.command()
@click.version_option()
def main() -> None:
    """Dynamic PET."""


if __name__ == "__main__":
    main(prog_name="dynamicpet")  # pragma: no cover
