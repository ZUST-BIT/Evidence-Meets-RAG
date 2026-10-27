"""Download the four exact EvidenceNet v10 inputs used by this project."""
import sys
from prepare_sources import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--only", "evidencenet", "--download"]
    main()
