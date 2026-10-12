#!/usr/bin/env python
"""Which cloud deployment these scripts talk to.

Every script in this directory reads somebody's BigQuery and writes somebody's GCS bucket, and
which somebody is not a detail the scripts should decide. So none of them carries a project id.
They ask here, and this asks the environment.

The variables are the same ones the runtime package reads, in `config.py`, so a clone that is
configured to serve is already configured to run these:

    export SMART_ROUTER_BQ_PROJECT=your-gcp-project
    export SMART_ROUTER_GCS_BUCKET=your-bucket

There is deliberately no fallback project. A default here is a script that quietly bills, reads
or deletes against whichever project the last maintainer happened to use, and the failure looks
like a permissions error rather than a configuration one. `--project` on the command line always
wins; with neither that nor the variable, argparse says which flag is missing and the help text
says which variable would have supplied it.

`SMART_ROUTER_BQ_DATASET` does have a default, because `smart_router` names nothing outside the
project it sits in and every deployment of this system uses it.
"""

from __future__ import annotations

import os

PROJECT_VAR = "SMART_ROUTER_BQ_PROJECT"
BUCKET_VAR = "SMART_ROUTER_GCS_BUCKET"
DATASET_VAR = "SMART_ROUTER_BQ_DATASET"

DEFAULT_DATASET = os.environ.get(DATASET_VAR) or "smart_router"

# Read once at import so `--help` works and so importing a script never raises. Either may be
# None, and the two `*_arg()` helpers below are what turn a None into a readable error.
#
# GOOGLE_CLOUD_PROJECT is the second chance because gcloud and the GCP client libraries already
# read it, so a machine that can talk to a project at all usually has it set. It is a fallback
# and not the name in the messages: the point of a router-specific variable is that you can aim
# these scripts somewhere other than your default project without exporting a lie.
DEFAULT_PROJECT = os.environ.get(PROJECT_VAR) or os.environ.get("GOOGLE_CLOUD_PROJECT")
DEFAULT_BUCKET = os.environ.get(BUCKET_VAR)


def project_arg() -> dict[str, object]:
    """Keyword arguments for a `--project` flag: defaulted from the environment, else required.

    Spelled as arguments to `add_argument` rather than as a value so that the missing case is
    argparse's to report. A script that resolved this itself would have to choose between
    raising at import, which breaks `--help` and breaks importing it from a test, and raising
    at first use, which is somewhere after the run has already started doing work.

        parser.add_argument("--project", **deployment.project_arg())
    """
    return _arg(DEFAULT_PROJECT, PROJECT_VAR, "GCP project to read and write")


def bucket_arg() -> dict[str, object]:
    """The same, for `--bucket`. Named separately because the bucket is not derivable."""
    return _arg(DEFAULT_BUCKET, BUCKET_VAR, "GCS bucket holding the router artifacts")


def require_project(value: str | None = None) -> str:
    """The project, for the few callers that resolve it outside argparse. Exits if unset."""
    return _require(value, DEFAULT_PROJECT, PROJECT_VAR)


def require_bucket(value: str | None = None) -> str:
    """The bucket, for the few callers that resolve it outside argparse. Exits if unset."""
    return _require(value, DEFAULT_BUCKET, BUCKET_VAR)


def _arg(found: str | None, variable: str, what: str) -> dict[str, object]:
    if found:
        return {"default": found, "help": f"{what}; defaults to ${variable}"}
    return {"required": True, "help": f"{what}; set ${variable} to default it"}


def _require(value: str | None, found: str | None, variable: str) -> str:
    resolved = value or found
    if not resolved:
        raise SystemExit(f"set {variable}, or pass it on the command line")
    return resolved
