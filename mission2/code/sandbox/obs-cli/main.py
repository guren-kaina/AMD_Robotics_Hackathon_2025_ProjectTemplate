import argparse
import os
import sys

from obsws_python import ReqClient
from obsws_python.error import OBSSDKError


def add_connection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--host",
        default=os.getenv("OBS_WS_HOST", "localhost"),
        help="OBS websocket host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("OBS_WS_PORT", "4455")),
        help="OBS websocket port (default: 4455)",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("OBS_WS_PASSWORD")
        or os.getenv("OBS_WEBSOCKET_PASSWORD", ""),
        help="OBS websocket password (fallback: OBS_WS_PASSWORD env)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("OBS_WS_TIMEOUT", "5")),
        help="Connection timeout seconds (default: 5)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control OBS via obs-websocket")
    add_connection_args(parser)

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_scenes = subparsers.add_parser("list-scenes", help="List scenes in OBS")
    list_scenes.set_defaults(func=handle_list_scenes)

    list_sources = subparsers.add_parser(
        "list-sources", help="List sources inside a scene"
    )
    list_sources.add_argument("scene", help="Target scene name")
    list_sources.set_defaults(func=handle_list_sources)

    switch_scene = subparsers.add_parser(
        "switch-scene", help="Switch the current program/preview scene"
    )
    switch_scene.add_argument("name", help="Scene name to switch to")
    switch_scene.add_argument(
        "--preview",
        action="store_true",
        help="Switch preview instead of program (Studio Mode)",
    )
    switch_scene.set_defaults(func=handle_switch_scene)

    source_visibility = subparsers.add_parser(
        "source-visibility", help="Show or hide a source inside a scene"
    )
    source_visibility.add_argument("scene", help="Scene name containing the source")
    source_visibility.add_argument("source", help="Source/input name to change")
    visibility_group = source_visibility.add_mutually_exclusive_group()
    visibility_group.add_argument(
        "--show", dest="enabled", action="store_true", help="Show/enable the source"
    )
    visibility_group.add_argument(
        "--hide", dest="enabled", action="store_false", help="Hide/disable the source"
    )
    source_visibility.set_defaults(func=handle_source_visibility, enabled=True)

    return parser


def handle_list_scenes(client: ReqClient, _: argparse.Namespace) -> int:
    response = client.get_scene_list()
    scenes = response.scenes or []
    current_program = getattr(response, "current_program_scene_name", None)
    current_preview = getattr(response, "current_preview_scene_name", None)

    if current_program:
        print(f"Program: {current_program}")
    if current_preview:
        print(f"Preview: {current_preview}")

    for scene in scenes:
        name = scene.get("sceneName") or scene.get("name") or "<unknown>"
        marker = ""
        if name == current_program:
            marker = " [PROGRAM]"
        elif current_preview and name == current_preview:
            marker = " [PREVIEW]"
        print(f"- {name}{marker}")

    return 0


def handle_list_sources(client: ReqClient, args: argparse.Namespace) -> int:
    response = client.get_scene_item_list(args.scene)
    items = getattr(response, "scene_items", []) or []

    if not items:
        print(f"No sources found in scene '{args.scene}'")
        return 0

    for item in items:
        name = item.get("sourceName") or item.get("name") or "<unknown>"
        scene_item_id = item.get("sceneItemId")
        enabled = item.get("sceneItemEnabled")
        locked = item.get("sceneItemLocked")
        flags = []
        if enabled is not None:
            flags.append("on" if enabled else "off")
        if locked is not None:
            flags.append("locked" if locked else "unlocked")
        flags_str = f" ({', '.join(flags)})" if flags else ""
        print(f"- {name} [id={scene_item_id}]{flags_str}")
    return 0


def handle_switch_scene(client: ReqClient, args: argparse.Namespace) -> int:
    if args.preview:
        client.set_current_preview_scene(args.name)
        print(f"Preview scene switched to '{args.name}'")
    else:
        client.set_current_program_scene(args.name)
        print(f"Program scene switched to '{args.name}'")
    return 0


def handle_source_visibility(client: ReqClient, args: argparse.Namespace) -> int:
    item = client.get_scene_item_id(args.scene, args.source)
    client.set_scene_item_enabled(args.scene, item.scene_item_id, args.enabled)
    state = "shown" if args.enabled else "hidden"
    print(
        f"Source '{args.source}' in scene '{args.scene}' set to {state} "
        f"(item id {item.scene_item_id})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        with ReqClient(
            host=args.host,
            port=args.port,
            password=args.password or "",
            timeout=args.timeout,
        ) as client:
            return args.func(client, args)
    except (OBSSDKError, ConnectionRefusedError, TimeoutError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
