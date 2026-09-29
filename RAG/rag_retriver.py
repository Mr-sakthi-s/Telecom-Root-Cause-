# ============================================================
# TELECOM RAG RETRIEVER
# ============================================================
#
# Responsibility:
#   1. Convert anonymized ML output using mapping_KB.txt
#   2. Retrieve relevant Telecom Knowledge
#   3. Retrieve relevant Historical Patterns
#   4. Return concise retrieval evidence
#
# This file DOES NOT generate RCA.
# RCA generation belongs to rca_engine.py.
#
# ============================================================

import os
import re
from typing import Dict, List, Any, Tuple

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

VECTOR_DB_DIR = os.path.join(
    BASE_DIR,
    "vector_db"
)

MAPPING_KB_PATH = os.path.join(
    DATA_DIR,
    "mapping_KB.txt"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================
#
# Prefer your local model cache.
# You can override this with:
#
# EMBEDDING_MODEL_PATH=...
#
# in .env
#

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL_PATH",
    r"C:\Users\sakthi murugan\.cache\huggingface\hub"
    r"\models--sentence-transformers--all-MiniLM-L6-v2"
    r"\snapshots\1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

KNOWLEDGE_TOP_K = 3

PATTERN_TOP_K = 5


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


# ============================================================
# KNOWLEDGE COLLECTION
# ============================================================

knowledge_db = Chroma(
    collection_name="telecom_knowledge",
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)


# ============================================================
# HISTORICAL PATTERN COLLECTION
# ============================================================

pattern_db = Chroma(
    collection_name="telecom_patterns",
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embeddings
)


# ============================================================
# MAPPING KB LOADER
# ============================================================

def load_mapping_kb() -> Dict[str, Any]:
    """
    Load the anonymized-ID → telecom-semantic mappings.

    Expected file:
        data/mapping_KB.txt
    """

    if not os.path.exists(
        MAPPING_KB_PATH
    ):

        raise FileNotFoundError(
            f"Mapping KB not found:\n"
            f"{MAPPING_KB_PATH}"
        )


    with open(
        MAPPING_KB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        lines = [
            line.strip()
            for line in file
            if line.strip()
        ]


    severity_map = {}
    resource_map = {}
    event_map = {}
    feature_ranges = []


    section = None
    current_id = None


    for line in lines:

        upper = line.upper()


        # ----------------------------------------------------
        # SECTION DETECTION
        # ----------------------------------------------------

        if "SEVERITY TYPE MAPPING" in upper:

            section = "severity"
            current_id = None
            continue


        if "RESOURCE TYPE MAPPING" in upper:

            section = "resource"
            current_id = None
            continue


        if "EVENT TYPE MAPPING" in upper:

            section = "event"
            current_id = None
            continue


        if "LOG FEATURE GROUPS" in upper:

            section = "feature"
            current_id = None
            continue


        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------

        if section == "severity":

            match = re.match(
                r"^(severity_type\s+\d+)$",
                line,
                flags=re.IGNORECASE
            )

            if match:

                current_id = match.group(1).lower()
                continue


            if (
                current_id
                and
                line.lower().startswith(
                    "category:"
                )
            ):

                severity_map[
                    current_id
                ] = line.split(
                    ":",
                    1
                )[1].strip()

                continue


        # ----------------------------------------------------
        # RESOURCE
        # ----------------------------------------------------

        if section == "resource":

            match = re.match(
                r"^(resource_type\s+\d+)$",
                line,
                flags=re.IGNORECASE
            )

            if match:

                current_id = match.group(1).lower()
                continue


            if (
                current_id
                and
                line.lower().startswith(
                    "category:"
                )
            ):

                resource_map[
                    current_id
                ] = line.split(
                    ":",
                    1
                )[1].strip()

                continue


        # ----------------------------------------------------
        # EVENT
        # ----------------------------------------------------

        if section == "event":

            match = re.match(
                r"^(event_type\s+\d+)$",
                line,
                flags=re.IGNORECASE
            )

            if match:

                current_id = match.group(1).lower()
                continue


            if (
                current_id
                and
                not line.startswith("=")
            ):

                event_map[
                    current_id
                ] = line.strip()

                continue


        # ----------------------------------------------------
        # FEATURE RANGE
        # ----------------------------------------------------

        if section == "feature":

            match = re.match(
                r"^feature\s+(\d+)(?:-(\d+))?$",
                line,
                flags=re.IGNORECASE
            )

            if match:

                start = int(
                    match.group(1)
                )

                end = int(
                    match.group(2)
                ) if match.group(2) else start


                feature_ranges.append(
                    {
                        "start": start,
                        "end": end,
                        "group": None
                    }
                )

                continue


            if (
                feature_ranges
                and
                feature_ranges[-1]["group"] is None
            ):

                feature_ranges[-1][
                    "group"
                ] = line.strip()


    return {

        "severity":
            severity_map,

        "resource":
            resource_map,

        "event":
            event_map,

        "feature":
            feature_ranges
    }


# ============================================================
# LOAD MAPPINGS ONCE
# ============================================================

MAPPING = load_mapping_kb()


# ============================================================
# MAP FEATURE
# ============================================================

def map_feature(
    feature_id: str
) -> str:

    match = re.search(
        r"(\d+)",
        str(feature_id)
    )

    if not match:

        return "Unknown Feature Group"


    feature_number = int(
        match.group(1)
    )


    for item in MAPPING[
        "feature"
    ]:

        if (
            item["start"]
            <= feature_number
            <= item["end"]
        ):

            return item["group"] or (
                "Unknown Feature Group"
            )


    return "Unknown Feature Group"


# ============================================================
# MAP ML OUTPUT
# ============================================================

def map_ml_output(
    ml_output: Dict[str, Any]
) -> Dict[str, Any]:

    severity_id = str(
        ml_output[
            "severity_type"
        ]
    ).lower()


    resource_id = str(
        ml_output[
            "resource_type"
        ]
    ).lower()


    event_ids = [
        str(event).lower()
        for event in
        ml_output[
            "event_types"
        ]
    ]


    feature_ids = [
        str(feature).lower()
        for feature in
        ml_output[
            "log_features"
        ]
    ]


    return {

        # ----------------------------------------------------
        # RAW
        # ----------------------------------------------------

        "raw_severity":
            ml_output[
                "severity_type"
            ],

        "raw_resource":
            ml_output[
                "resource_type"
            ],

        "raw_events":
            ml_output[
                "event_types"
            ],

        "raw_features":
            ml_output[
                "log_features"
            ],


        # ----------------------------------------------------
        # SEMANTIC
        # ----------------------------------------------------

        "severity":
            MAPPING[
                "severity"
            ].get(
                severity_id,
                "Unknown Severity"
            ),

        "resource":
            MAPPING[
                "resource"
            ].get(
                resource_id,
                "Unknown Resource"
            ),

        "events": [

            MAPPING[
                "event"
            ].get(
                event_id,
                "Unknown Event"
            )

            for event_id
            in event_ids
        ],

        "feature_groups": [

            map_feature(
                feature_id
            )

            for feature_id
            in feature_ids
        ],


        # ----------------------------------------------------
        # NUMERICAL
        # ----------------------------------------------------

        "predicted_fault_severity":
            ml_output[
                "predicted_fault_severity"
            ],

        "volume":
            ml_output[
                "volume"
            ]
    }


# ============================================================
# BUILD SEMANTIC QUERY
# ============================================================

def build_semantic_query(
    semantic_incident: Dict[str, Any]
) -> str:

    return f"""
Severity:
{semantic_incident['severity']}

Resource:
{semantic_incident['resource']}

Events:
{", ".join(semantic_incident['events'])}

Feature Groups:
{", ".join(semantic_incident['feature_groups'])}

Predicted Fault Severity:
{semantic_incident['predicted_fault_severity']}

Volume:
{semantic_incident['volume']}
""".strip()


# ============================================================
# BUILD PATTERN QUERY
# ============================================================

def build_pattern_query(
    ml_output: Dict[str, Any],
    semantic_incident: Dict[str, Any]
) -> str:

    raw_query = f"""
{ml_output['severity_type']}
{ml_output['resource_type']}
{", ".join(ml_output['event_types'])}
{", ".join(ml_output['log_features'])}
""".strip()


    semantic_query = (
        build_semantic_query(
            semantic_incident
        )
    )


    return (
        raw_query
        + "\n\n"
        + semantic_query
    )


# ============================================================
# RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(
    ml_output: Dict[str, Any]
) -> Dict[str, Any]:

    # --------------------------------------------------------
    # MAP ML OUTPUT
    # --------------------------------------------------------

    semantic_incident = (
        map_ml_output(
            ml_output
        )
    )


    # --------------------------------------------------------
    # QUERIES
    # --------------------------------------------------------

    semantic_query = (
        build_semantic_query(
            semantic_incident
        )
    )


    pattern_query = (
        build_pattern_query(
            ml_output,
            semantic_incident
        )
    )


    # --------------------------------------------------------
    # KNOWLEDGE RETRIEVAL
    # --------------------------------------------------------

    knowledge_results = (
        knowledge_db
        .similarity_search_with_relevance_scores(
            semantic_query,
            k=KNOWLEDGE_TOP_K
        )
    )


    # --------------------------------------------------------
    # PATTERN RETRIEVAL
    # --------------------------------------------------------

    pattern_results = (
        pattern_db
        .similarity_search_with_relevance_scores(
            pattern_query,
            k=PATTERN_TOP_K
        )
    )


    return {

        "semantic_incident":
            semantic_incident,

        "semantic_query":
            semantic_query,

        "knowledge":
            knowledge_results,

        "patterns":
            pattern_results
    }


# ============================================================
# CLEAN DOCUMENT PREVIEW
# ============================================================

def clean_document(
    text: str,
    max_chars: int = 500
) -> str:

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text.strip()
    )

    if len(text) > max_chars:

        return (
            text[:max_chars].rstrip()
            + "..."
        )

    return text


# ============================================================
# PRINT CONCISE RETRIEVAL
# ============================================================

def print_retrieval(
    result: Dict[str, Any]
):

    semantic = (
        result[
            "semantic_incident"
        ]
    )


    # ========================================================
    # INCIDENT INTERPRETATION
    # ========================================================

    print("\n")
    print("=" * 80)
    print("INCIDENT INTERPRETATION")
    print("=" * 80)

    print(
        f"Severity       : "
        f"{semantic['severity']}"
    )

    print(
        f"Resource       : "
        f"{semantic['resource']}"
    )

    print(
        f"Events         : "
        f"{', '.join(semantic['events'])}"
    )

    print(
        f"Feature Groups : "
        f"{', '.join(semantic['feature_groups'])}"
    )

    print(
        f"Volume         : "
        f"{semantic['volume']}"
    )


    # ========================================================
    # KNOWLEDGE
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TOP TELECOM KNOWLEDGE")
    print("=" * 80)


    knowledge = result[
        "knowledge"
    ]


    if not knowledge:

        print(
            "No knowledge documents found."
        )

    else:

        for index, (
            doc,
            score
        ) in enumerate(
            knowledge,
            start=1
        ):

            print("\n")
            print(
                f"[{index}] "
                f"Relevance: "
                f"{score:.3f}"
            )

            print(
                clean_document(
                    doc.page_content
                )
            )


    # ========================================================
    # HISTORICAL PATTERNS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("TOP HISTORICAL PATTERNS")
    print("=" * 80)


    patterns = result[
        "patterns"
    ]


    if not patterns:

        print(
            "No historical patterns found."
        )

    else:

        for index, (
            doc,
            score
        ) in enumerate(
            patterns,
            start=1
        ):

            print("\n")
            print(
                f"[{index}] "
                f"Relevance: "
                f"{score:.3f}"
            )

            print(
                clean_document(
                    doc.page_content
                )
            )


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Test ML output
    # --------------------------------------------------------

    ml_output = {

        "predicted_fault_severity":
            2,

        "severity_type":
            "severity_type 2",

        "resource_type":
            "resource_type 5",

        "event_types": [

            "event_type 10",

            "event_type 12",

            "event_type 14"
        ],

        "log_features": [

            "feature 64",

            "feature 82",

            "feature 91"
        ],

        "volume":
            250
    }


    # --------------------------------------------------------
    # Collection check
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("RAG RETRIEVER")
    print("=" * 80)


    print(
        f"Knowledge documents: "
        f"{knowledge_db._collection.count()}"
    )

    print(
        f"Pattern documents: "
        f"{pattern_db._collection.count()}"
    )


    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    result = retrieve_documents(
        ml_output
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_retrieval(
        result
    )