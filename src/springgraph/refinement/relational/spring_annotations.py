"""Spring annotation classification helpers."""

COMPONENT_ANNOTATIONS = {
    "SpringBootApplication",
    "RestController",
    "Controller",
    "Service",
    "Repository",
    "Component",
    "Configuration",
}

ROUTE_ANNOTATIONS = {
    "RequestMapping",
    "GetMapping",
    "PostMapping",
    "PutMapping",
    "DeleteMapping",
    "PatchMapping",
}

IGNORED_EXTERNAL_ANNOTATIONS = {
    "Autowired",
    "PathVariable",
    "RequestBody",
    "RequestParam",
    "ResponseBody",
    "Override",
}


def is_component_annotation(name: str) -> bool:
    """Return whether an annotation marks a Spring component."""
    return name in COMPONENT_ANNOTATIONS


def is_route_annotation(name: str) -> bool:
    """Return whether an annotation maps an HTTP route."""
    return name in ROUTE_ANNOTATIONS


def http_method_for_annotation(name: str) -> str:
    """Return HTTP method implied by a Spring route annotation."""
    return {
        "GetMapping": "GET",
        "PostMapping": "POST",
        "PutMapping": "PUT",
        "DeleteMapping": "DELETE",
        "PatchMapping": "PATCH",
        "RequestMapping": "ANY",
    }.get(name, "ANY")


def should_ignore_annotation_ref(name: str) -> bool:
    """Return whether a framework annotation should not become unresolved."""
    return (
        name in COMPONENT_ANNOTATIONS
        or name in ROUTE_ANNOTATIONS
        or name in IGNORED_EXTERNAL_ANNOTATIONS
    )
