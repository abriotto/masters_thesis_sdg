#!/usr/bin/env bash
# Build an isolated venv for the Knaebel discopy shallow discourse parser.
# Kept completely separate from the project's sdglogs env: discopy needs
# numpy<2 + TensorFlow, which would break torch 2.11 / numpy 2.2 in sdglogs.
set -x

# The venv is created first so the CA bundle can live beside it.
# Avast Web/Mail Shield re-signs all TLS traffic with its own root CA. The
# venv's bundled pip verifies against its vendored certifi bundle, which does
# not contain that root, so every index request fails. This bundle is exported
# from the machine's own Windows trust store, which already trusts the Avast
# root - certificate verification stays fully enabled.
CA="$VENV/win-ca-bundle.pem"
export PIP_CERT="$CA"
export SSL_CERT_FILE="$CA"
export REQUESTS_CA_BUNDLE="$CA"
export CURL_CA_BUNDLE="$CA"
export GIT_SSL_CAINFO="$CA"
VENV="C:/Users/annab/discopy-env"
# Sources live INSIDE the venv, next to the checkpoint, because these are
# editable installs: a temp directory made the installed package vanish
# when the directory was cleaned, leaving a venv that imported nothing.
SRC="$VENV/src"

"C:/Users/annab/miniconda3/envs/sdglogs/python.exe" -m venv "$VENV" || exit 1
P="$VENV/Scripts/python.exe"
"$P" -V || exit 1

"$P" -m pip install --upgrade pip setuptools wheel || exit 1

# Pinned numeric stack: TF 2.10.1 is the last TensorFlow with native Windows
# support and is the newest that still accepts numpy<1.24 / protobuf<3.20.
"$P" -m pip install "numpy==1.23.5" "protobuf==3.19.6" "tensorflow==2.10.1" || exit 1

# discopy pins transformers==4.8.0, whose tokenizers==0.10.x has no cp310
# wheel. The transformers surface discopy_data actually uses is
# AutoTokenizer / TFAutoModel / tokenize / convert_tokens_to_ids /
# build_inputs_with_special_tokens / model(output_hidden_states=True), all
# unchanged since 4.x, so a cp310-installable release is used instead.
"$P" -m pip install "transformers==4.30.2" || exit 1

# setup.py lists the dead `sklearn` shim, which now hard-fails on install, so
# the real dependencies go in by hand and discopy itself goes in --no-deps.
"$P" -m pip install "nltk>=3.4" joblib scikit-learn sklearn-crfsuite click tqdm || exit 1

mkdir -p "$SRC"
git clone --quiet https://github.com/rknaebel/discopy-data "$SRC/discopy-data" || exit 1
git clone --quiet https://github.com/rknaebel/discopy "$SRC/discopy" || exit 1
git -C "$SRC/discopy" checkout --quiet 1.1.0 || exit 1

echo "=== discopy-data HEAD ==="; git -C "$SRC/discopy-data" rev-parse HEAD
echo "=== discopy 1.1.0 HEAD ==="; git -C "$SRC/discopy" rev-parse HEAD

"$P" -m pip install --no-deps -e "$SRC/discopy-data" || exit 1
"$P" -m pip install --no-deps -e "$SRC/discopy" || exit 1

"$P" -c "import nltk; nltk.download('punkt')" || true

echo "=== INSTALLED ==="
"$P" -m pip list | grep -iE "tensorflow|transformers|tokenizers|numpy|nltk|discopy|scikit|crfsuite|protobuf"

# The CA bundle lives at $VENV/win-ca-bundle.pem. Rebuild it by exporting the
# Windows trust store BEFORE the first pip call:
#   python -c "import ssl,base64;open(r'C:/Users/annab/discopy-env/win-ca-bundle.pem','w').write(''.join('-----BEGIN CERTIFICATE-----\n'+base64.encodebytes(c).decode()+'-----END CERTIFICATE-----\n' for c,e,t in ssl.enum_certificates('ROOT')+ssl.enum_certificates('CA') if e=='x509_asn'))"
# This is needed only because Avast Web/Mail Shield re-signs TLS traffic; the
# bundle comes from the machine's own trust store, so verification stays on.
#
# The parser itself needs no network: bert-base-cased is already in the local
# HuggingFace cache, and running with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# both avoids the TLS interception and guarantees the same tokenizer snapshot
# the base corpus was parsed with.
#
#   HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
#   "C:/Users/annab/discopy-env/Scripts/python.exe" \
#     src/justification_analysis/discopy_parser/run_discopy_on_justifications.py \
#     --model-path "C:/Users/annab/discopy-env/models/bert-codi" --stage <stage>
