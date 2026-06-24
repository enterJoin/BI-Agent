"""Java naming helpers."""


def package_name(source: str) -> str | None:
    """Extract a package declaration from Java source."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("package ") and stripped.endswith(";"):
            return stripped.removeprefix("package ").removesuffix(";").strip()
    return None


def class_fqn(package: str | None, class_name: str) -> str:
    """Build a fully qualified class name."""
    return f"{package}.{class_name}" if package else class_name


def member_fqn(owner_qualified_name: str, member_name: str) -> str:
    """Build a fully qualified member name."""
    return f"{owner_qualified_name}.{member_name}"


def simple_name(qualified_name: str) -> str:
    """Return the last segment of a Java qualified name."""
    return qualified_name.rsplit(".", maxsplit=1)[-1]
