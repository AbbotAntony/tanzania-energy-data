"""Command-line interface.

Examples (PowerShell, from the project folder)::

    .\\.venv\\Scripts\\python.exe -m pipeline list-sources
    .\\.venv\\Scripts\\python.exe -m pipeline run --source ewura
    .\\.venv\\Scripts\\python.exe -m pipeline run --source ewura --only list
    .\\.venv\\Scripts\\python.exe -m pipeline qa
"""

from __future__ import annotations

import argparse
import logging
import sys

from pipeline.http_client import HttpClient
from pipeline.logging_setup import setup_logging

log = logging.getLogger("pipeline")


def _cmd_run(args: argparse.Namespace) -> int:
    from scrapers import REGISTRY

    if args.source not in REGISTRY:
        print(f"unknown source {args.source!r}; choose from {sorted(REGISTRY)}")
        return 2
    log_file = setup_logging(args.source, verbose=args.verbose)
    log.info("logging to %s", log_file)
    cls = REGISTRY[args.source]
    scraper = cls(HttpClient(refresh=args.refresh, system_trust=cls.system_trust))
    refs = scraper.list_documents()
    log.info("%d documents listed", len(refs))
    if args.only == "list":
        for r in refs:
            print(f"{r.category:20} {r.pub_date or '':10}  {r.title}")
        return 0
    docs = scraper.download(refs)
    log.info("%d documents stored", len(docs))
    if args.only != "download":
        scraper.export(scraper.parse(docs))
    scraper._report_failures()
    return 1 if scraper.failures else 0


def _cmd_list_sources(_: argparse.Namespace) -> int:
    from scrapers import REGISTRY

    for name, cls in sorted(REGISTRY.items()):
        print(f"{name:12} {(cls.__doc__ or '').strip().splitlines()[0]}")
    return 0


def _cmd_qa(args: argparse.Namespace) -> int:
    from pipeline import qa

    setup_logging("qa", verbose=args.verbose)
    qa.run_all()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="scrape, parse and export one source")
    run.add_argument("--source", required=True)
    run.add_argument("--refresh", action="store_true", help="ignore the HTTP cache")
    run.add_argument("--only", choices=("list", "download", "parse"), default=None,
                     help="stop after this step (parse = parse+export)")
    run.set_defaults(func=_cmd_run)
    sub.add_parser("list-sources").set_defaults(func=_cmd_list_sources)
    sub.add_parser("qa", help="missing-year and cross-source checks").set_defaults(func=_cmd_qa)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
