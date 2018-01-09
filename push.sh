#!/usr/bin/bash
rsync -avrz \
    --include '*/' \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --include '*.py' \
    --exclude '*' \
    -e 'ssh -p 58022' \
    * w330@atcremers75.informatik.tu-muenchen.de:Projekte/GO_DILab
    # * hikaru@10.155.208.160:Projekte/GO_DILab
    