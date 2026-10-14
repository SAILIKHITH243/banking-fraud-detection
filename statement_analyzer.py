
"""Multi-format statement analyzer: PDF, OCR, CSV, TXT, Excel and images."""

import io
import os
import re
import shutil

import pandas as pd

OUTPUT_COLUMNS = [
    "date", "description", "debit", "credit", "amount", "currency",
    "balance", "balance_direction", "direction", "transaction_id",
    "category", "source_format", "review_reason"
]

DATE_PART = (
    r"(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{1,2}\s+[A-Za-z]{3,9},?\s+\d{4}|"
    r"[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4})"
)

DATE_LINE = re.compile(
    r"^\s*(?:(\d{1,5})\s+)?(" +
    DATE_PART +
    r")(?:[,;\s]+(.*))?$",
    re.I
)

MONEY = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?P<cur>₹|INR\b|Rs\.?\s*|\$|USD\b|€|EUR\b|£|GBP\b)?\s*"
    r"(?P<sign>[+-])?\s*"
    r"(?P<num>\d[\d,]*(?:\.\d{1,2})?)"
    r"\s*(?:\(\s*(?P<dc1>Dr|Cr)\s*\)|(?P<dc2>Dr|Cr)\b)?"
    r"(?![A-Za-z0-9])",
    re.I
)

ALIASES = {
    "date": [
        "transaction date", "date time", "transaction datetime",
        "txn date", "value date", "timestamp", "date"
    ],
    "description": [
        "transaction description", "transaction details",
        "payment details", "particulars", "narration", "remarks",
        "description", "details", "merchant", "payee",
        "recipient", "sender", "beneficiary"
    ],
    "debit": [
        "debit amount", "amount debited", "withdrawal amount",
        "withdrawals", "debit", "money out", "sent amount"
    ],
    "credit": [
        "credit amount", "amount credited", "deposit amount",
        "deposits", "credit", "money in", "received amount"
    ],
    "amount": [
        "transaction amount", "amount inr", "amount rs",
        "amount", "transaction value", "value"
    ],
    "balance": [
        "closing balance", "available balance",
        "running balance", "balance"
    ],
    "direction": [
        "transaction type", "dr cr", "debit credit",
        "direction", "type"
    ],
    "transaction_id": [
        "transaction id", "transaction reference",
        "reference number", "reference id", "txn id",
        "upi transaction id", "utr", "rrn", "id"
    ],
    "currency": ["currency code", "currency"],
    "status": ["transaction status", "payment status", "status"]
}


def extract_amount(value):
    """Convert a monetary value to a number when possible."""
    if value is None:
        return None

    if not isinstance(value, str) and pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().replace("−", "-")

    if not text or text.lower() in {
        "nan", "none", "-", "--", "n/a", "na"
    }:
        return None

    negative_parentheses = (
        text.startswith("(") and text.endswith(")")
    )

    text = re.sub(r"[^0-9,.+\-]", "", text)

    if not text or text in {"-", "+", "."}:
        return None

    try:
        number = float(text.replace(",", ""))
    except ValueError:
        return None

    return -abs(number) if negative_parentheses else number


def categorize_transaction(description):
    text = str(description or "").lower()

    rules = {
        "Food & Dining": [
            "restaurant", "food", "swiggy", "zomato", "cafe",
            "grocery", "groceries", "instamart", "blinkit"
        ],
        "Transport": [
            "uber", "ola", "rapido", "fuel", "petrol",
            "diesel", "metro", "irctc", "railway", "flight", "bus"
        ],
        "Shopping": [
            "amazon", "flipkart", "myntra", "shopping",
            "retail", "mall"
        ],
        "Bills & Utilities": [
            "electricity", "water bill", "gas bill",
            "broadband", "recharge", "mobile bill", "jio", "airtel"
        ],
        "Healthcare": [
            "hospital", "pharmacy", "medical",
            "clinic", "doctor", "medicine"
        ],
        "Entertainment": [
            "netflix", "spotify", "movie", "cinema",
            "bookmyshow", "prime video", "gaming"
        ],
        "Education": [
            "school", "college", "university", "course",
            "tuition", "exam fee"
        ],
        "Rent & Housing": [
            "rent", "landlord", "maintenance", "housing"
        ],
        "Refunds & Reversals": [
            "refund", "reversal", "chargeback", "cashback"
        ],
        "Income": [
            "salary", "payroll", "interest credited",
            "dividend", "stipend"
        ],
        "Bank Charges": [
            "bank charge", "service charge", "sms charge",
            "gst on charge", "atm fee"
        ],
        "Transfer": [
            "upi", "imps", "neft", "rtgs", "transfer",
            "fund transfer"
        ]
    }

    for category, keywords in rules.items():
        if any(word in text for word in keywords):
            return category

    return "Other"


def detect_statement_type(text):
    text = str(text or "").lower()

    providers = [
        (["phonepe", "phone pe"], "PhonePe"),
        (["google pay", "gpay"], "Google Pay"),
        (["union bank"], "Union Bank"),
        (["state bank of india", "sbi"], "SBI"),
        (["hdfc bank"], "HDFC Bank"),
        (["icici bank"], "ICICI Bank"),
        (["axis bank"], "Axis Bank")
    ]

    for keywords, provider in providers:
        if any(word in text for word in keywords):
            return provider

    if "upi" in text:
        return "UPI statement"

    if any(word in text for word in [
        "account statement", "statement period",
        "opening balance", "closing balance"
    ]):
        return "Bank statement (provider unknown)"

    return "Unknown financial statement"


def _txt(value):
    if value is None:
        return ""

    if not isinstance(value, str) and pd.isna(value):
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def _direction(text):
    """Infer direction only from recognizable words."""
    text = str(text or "").lower()

    credit_words = (
        r"\b(refund(?:ed)?|reversal|cashback received|"
        r"payment received|money received|received from|"
        r"credited|deposit|credit|cr)\b"
    )

    debit_words = (
        r"\b(paid|paid to|sent to|money sent|withdraw(?:al)?|"
        r"debit|dr|purchase|payment to)\b"
    )

    if re.search(credit_words, text):
        return "Credit"

    if re.search(debit_words, text):
        return "Debit"

    return "Unknown"


def _currency(text):
    text = str(text or "")

    currencies = [
        (r"₹|\bINR\b|\bRs\.?", "INR"),
        (r"\$|\bUSD\b", "USD"),
        (r"€|\bEUR\b", "EUR"),
        (r"£|\bGBP\b", "GBP")
    ]

    for pattern, code in currencies:
        if re.search(pattern, text, re.I):
            return code

    return "Unknown"


def _date(value):
    if value is None:
        return pd.NaT

    try:
        return pd.to_datetime(
            str(value).strip(),
            dayfirst=True,
            errors="coerce"
        )
    except Exception:
        return pd.NaT


def _record(
    date,
    description,
    amount=None,
    debit=None,
    credit=None,
    balance=None,
    balance_dir="Unknown",
    direction=None,
    txid="",
    currency="Unknown",
    source="PDF",
    review=""
):
    parsed_date = _date(date)
    description = _txt(description) or "Unspecified transaction"

    amount, debit, credit, balance = map(
        extract_amount, [amount, debit, credit, balance]
    )

    if debit is not None and debit != 0:
        debit = abs(debit)
        credit = 0.0
        amount = debit
        direction = "Debit"

    elif credit is not None and credit != 0:
        credit = abs(credit)
        debit = 0.0
        amount = credit
        direction = "Credit"

    else:
        direction = direction or _direction(description)

        if amount is not None and amount < 0:
            direction = "Debit"
            amount = abs(amount)

        if amount is not None and direction == "Debit":
            debit, credit = abs(amount), 0.0

        elif amount is not None and direction == "Credit":
            debit, credit = 0.0, abs(amount)

        else:
            debit = credit = None

    if amount is None:
        if debit not in (None, 0):
            amount = abs(debit)
        elif credit not in (None, 0):
            amount = abs(credit)

    if direction not in {"Debit", "Credit"}:
        direction = direction or _direction(description)

    if pd.isna(parsed_date):
        review = (review + "; " if review else "") + "Date unclear"

    if amount is None or amount <= 0:
        review = (review + "; " if review else "") + "Amount unclear"

    if direction == "Unknown":
        review = (
            (review + "; " if review else "")
            + "Debit/credit direction unclear"
        )

    return {
        "date": parsed_date,
        "description": description,
        "debit": debit,
        "credit": credit,
        "amount": abs(amount) if amount is not None else None,
        "currency": (
            currency if currency != "Unknown"
            else _currency(description)
        ),
        "balance": balance,
        "balance_direction": balance_dir,
        "direction": direction,
        "transaction_id": _txt(txid),
        "category": categorize_transaction(description),
        "source_format": source,
        "review_reason": review
    }


def _norm_header(value):
    value = _txt(value).lower().replace("₹", " inr ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _find_col(headers, aliases):
    normalized = [_norm_header(item) for item in headers]

    for alias in aliases:
        target = _norm_header(alias)

        for index, header in enumerate(normalized):
            if header == target:
                return index

    for alias in sorted(aliases, key=len, reverse=True):
        target = _norm_header(alias)

        for index, header in enumerate(normalized):
            if target and target in header:
                return index

    return None


def _table_records(table):
    """Read transaction tables whose column headings can be identified."""
    if not table or len(table) < 2:
        return []

    mapping = None
    header_index = None

    for index, row in enumerate(table[:7]):
        candidate = {
            field: _find_col(row, aliases)
            for field, aliases in ALIASES.items()
        }

        candidate = {
            field: col for field, col in candidate.items()
            if col is not None
        }

        if (
            "date" in candidate
            and any(
                field in candidate
                for field in ("amount", "debit", "credit")
            )
        ):
            mapping = candidate
            header_index = index
            break

    if mapping is None:
        return []

    results = []

    for row in table[header_index + 1:]:
        if not row or not any(_txt(cell) for cell in row):
            continue

        def cell(field):
            col = mapping.get(field)
            return row[col] if col is not None and col < len(row) else None

        date_value = cell("date")

        if pd.isna(_date(date_value)):
            continue

        description = _txt(cell("description"))
        amount_text = _txt(cell("amount"))
        balance_text = _txt(cell("balance"))

        direction_text = _txt(cell("direction"))
        status = _txt(cell("status"))

        debit = extract_amount(cell("debit"))
        credit = extract_amount(cell("credit"))
        amount = extract_amount(amount_text)
        balance = extract_amount(balance_text)

        amount_marker = re.search(
            r"\b(Dr|Cr)\b", amount_text, re.I
        )
        balance_marker = re.search(
            r"\b(Dr|Cr)\b", balance_text, re.I
        )

        if amount_marker:
            direction = (
                "Debit"
                if amount_marker.group(1).lower() == "dr"
                else "Credit"
            )
        else:
            direction = _direction(
                " ".join([
                    description, direction_text, status
                ])
            )

        currency = (
            _txt(cell("currency")).upper()
            or _currency(" ".join(_txt(c) for c in row))
        )

        record = _record(
            date_value,
            description or direction_text or "Statement transaction",
            amount,
            debit,
            credit,
            balance,
            balance_marker.group(1).title()
            if balance_marker else "Unknown",
            direction,
            cell("transaction_id") or "",
            currency,
            "PDF table"
        )

        if re.search(
            r"failed|cancelled|canceled|declined|pending",
            status,
            re.I
        ):
            record["review_reason"] = (
                record["review_reason"] + "; "
                if record["review_reason"] else ""
            ) + "Payment status: " + status

        results.append(record)

    return results


def _money_candidates(text):
    results = []

    for match in MONEY.finditer(text):
        number = match.group("num")
        currency = match.group("cur") or ""
        sign = match.group("sign") or ""
        marker = (
            match.group("dc1")
            or match.group("dc2")
            or ""
        ).lower()

        # Do not treat ordinary reference numbers as money.
        if not (
            currency or "." in number or sign or marker
        ):
            continue

        value = extract_amount(number)

        if value is None:
            continue

        if sign == "-":
            value = -abs(value)

        results.append({
            "start": match.start(),
            "end": match.end(),
            "raw": match.group(0).strip(),
            "value": value,
            "dc": marker,
            "currency": _currency(currency)
        })

    return results


def parse_statement_text(text):
    """General fallback parser for date-led statement text."""
    lines = [
        line.strip()
        for line in str(text or "").replace("\x00", " ").splitlines()
        if line.strip()
    ]

    starts = [
        index for index, line in enumerate(lines)
        if DATE_LINE.match(line)
        or re.match(r"^[•*\-]?\s*" + DATE_PART, line, re.I)
    ]

    records = []

    for position, line_index in enumerate(starts):
        end_index = (
            starts[position + 1]
            if position + 1 < len(starts)
            else len(lines)
        )

        match = DATE_LINE.match(lines[line_index])

        if not match:
            continue

        date_value = match.group(2)
        remainder = match.group(3) or ""
        transaction_id = ""

        id_match = re.match(
            r"^\s*([A-Z0-9-]{8,})\s+(.*)$",
            remainder,
            re.I
        )

        if id_match:
            transaction_id = id_match.group(1)
            remainder = id_match.group(2)

        block = re.sub(
            r"\s+",
            " ",
            (remainder + " " + " ".join(
                lines[line_index + 1:end_index]
            )).strip()
        )[:2500]

        candidates = _money_candidates(block)

        if not candidates:
            continue

        explicit = [item for item in candidates if item["dc"]]

        # Many bank statements place amount before balance.
        if len(explicit) >= 2:
            chosen = explicit[-2]
            balance_token = explicit[-1]
        elif len(candidates) >= 2:
            chosen = candidates[-2]
            balance_token = candidates[-1]
        else:
            chosen = candidates[0]
            balance_token = None

        if chosen["dc"] == "dr":
            direction = "Debit"
        elif chosen["dc"] == "cr":
            direction = "Credit"
        elif chosen["value"] < 0:
            direction = "Debit"
        else:
            cleaned_block = block

            for candidate in candidates:
                cleaned_block = cleaned_block.replace(
                    candidate["raw"], " ", 1
                )

            direction = _direction(cleaned_block)

        description = block

        for candidate in sorted(
            candidates, key=lambda item: item["start"], reverse=True
        ):
            description = (
                description[:candidate["start"]]
                + " "
                + description[candidate["end"]:]
            )

        description = re.sub(
            r"\b(?:DR|CR|DEBIT|CREDIT|SUCCESSFUL|SUCCESS|"
            r"COMPLETED|TRANSACTION)\b",
            " ",
            description,
            flags=re.I
        )

        if transaction_id:
            description = description.replace(transaction_id, " ")

        description = re.sub(
            r"\s+", " ", description
        ).strip(" |:-,;.") or "Statement transaction"

        balance = (
            abs(balance_token["value"])
            if balance_token else None
        )

        balance_direction = (
            balance_token["dc"].title()
            if balance_token and balance_token["dc"]
            else "Unknown"
        )

        review = ""

        if direction == "Unknown":
            review = "Payment direction unclear; verify before totals"

        if len(candidates) > 2:
            review = (
                (review + "; " if review else "")
                + "Multiple monetary values found; verify amount and balance"
            )

        amount = abs(chosen["value"])

        records.append(_record(
            date_value,
            description,
            amount,
            amount if direction == "Debit" else (
                0 if direction == "Credit" else None
            ),
            amount if direction == "Credit" else (
                0 if direction == "Debit" else None
            ),
            balance,
            balance_direction,
            direction,
            transaction_id,
            chosen["currency"],
            "PDF/text/OCR",
            review
        ))

    return pd.DataFrame(records, columns=OUTPUT_COLUMNS)


def _column_records(table, source):
    """Import CSV/Excel columns by recognizing their header names."""
    if table is None or table.empty:
        return []

    def find_mapping(headers):
        result = {
            field: _find_col(headers, aliases)
            for field, aliases in ALIASES.items()
        }
        return {
            field: index for field, index in result.items()
            if index is not None
        }

    working = table
    mapping = find_mapping(list(working.columns))

    if (
        "date" not in mapping
        or not any(
            field in mapping
            for field in ("amount", "debit", "credit")
        )
    ):
        for header_row in range(min(8, len(table))):
            headers = [
                _txt(value)
                for value in table.iloc[header_row].tolist()
            ]
            candidate = find_mapping(headers)

            if (
                "date" in candidate
                and any(
                    field in candidate
                    for field in ("amount", "debit", "credit")
                )
            ):
                working = table.iloc[header_row + 1:].copy()
                working.columns = headers
                mapping = candidate
                break
        else:
            return []

    results = []

    for _, row in working.iterrows():
        def cell(field):
            index = mapping.get(field)
            return row.iloc[index] if index is not None else None

        date_value = cell("date")

        if pd.isna(_date(date_value)):
            continue

        description = _txt(cell("description"))
        amount = extract_amount(cell("amount"))
        debit = extract_amount(cell("debit"))
        credit = extract_amount(cell("credit"))
        balance = extract_amount(cell("balance"))

        direction_text = " ".join([
            description, _txt(cell("direction")), _txt(cell("status"))
        ])
        direction = _direction(direction_text)

        if debit is not None and debit != 0:
            direction, amount, credit = "Debit", abs(debit), 0.0
        elif credit is not None and credit != 0:
            direction, amount, debit = "Credit", abs(credit), 0.0
        elif amount is not None:
            if amount < 0:
                direction, amount = "Debit", abs(amount)

            if direction == "Debit":
                debit, credit = amount, 0.0
            elif direction == "Credit":
                debit, credit = 0.0, amount
            else:
                debit = credit = None

        currency = (
            _txt(cell("currency")).upper()
            or _currency(" ".join([description, _txt(cell("amount"))]))
        )

        record = _record(
            date_value,
            description or "Imported transaction",
            amount,
            debit,
            credit,
            balance,
            direction=direction,
            txid=cell("transaction_id") or "",
            currency=currency,
            source=source
        )

        status = _txt(cell("status"))

        if re.search(
            r"failed|cancelled|canceled|declined|pending",
            status,
            re.I
        ):
            record["review_reason"] = (
                record["review_reason"] + "; "
                if record["review_reason"] else ""
            ) + "Payment status: " + status

        results.append(record)

    return results


def _read_table(uploaded_file, extension):
    data = uploaded_file.getvalue()

    if not data:
        raise ValueError("The uploaded file is empty.")

    if extension in {".csv", ".txt"}:
        last_error = None

        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                return pd.read_csv(
                    io.BytesIO(data),
                    sep=None,
                    engine="python",
                    encoding=encoding
                )
            except Exception as error:
                last_error = error

        raise ValueError(f"Could not read CSV/TXT: {last_error}")

    try:
        return pd.read_excel(io.BytesIO(data))
    except Exception as error:
        raise ValueError(
            f"Could not read Excel file: {error}. "
            "Install openpyxl for XLSX or xlrd for XLS."
        ) from error


def _tesseract_setup():
    import pytesseract

    if not shutil.which("tesseract"):
        locations = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
        ]

        for path in locations:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break

    return pytesseract


def _ocr_image_bytes(data):
    try:
        from PIL import Image

        pytesseract = _tesseract_setup()
        image = Image.open(io.BytesIO(data)).convert("RGB")

        try:
            return pytesseract.image_to_string(
                image,
                lang="eng",
                config="--oem 3 --psm 6"
            )
        finally:
            image.close()

    except Exception as error:
        raise ValueError(
            f"Image OCR failed. Check Tesseract and Pillow: {error}"
        ) from error


def _pdf_content(data, password=None):
    import pdfplumber

    page_texts = []
    tables = []

    try:
        with pdfplumber.open(
            io.BytesIO(data),
            password=password or ""
        ) as pdf:
            for page in pdf.pages:
                page_texts.append(page.extract_text() or "")

                try:
                    tables.extend(page.extract_tables() or [])
                except Exception:
                    pass

    except Exception as error:
        message = str(error).lower()

        if any(word in message for word in [
            "password", "encrypt", "decrypt"
        ]):
            raise ValueError(
                "This PDF needs the correct password."
            ) from error

        raise ValueError(f"Could not open PDF: {error}") from error

    table_records = []

    for table in tables:
        table_records.extend(_table_records(table))

    native_df = parse_statement_text("\n".join(page_texts))
    native_records = (
        native_df.to_dict("records")
        if not native_df.empty else []
    )

    # If regular text extraction fails, OCR the scanned pages.
    if not table_records and not native_records:
        try:
            import pymupdf
            from PIL import Image

            pytesseract = _tesseract_setup()
            document = pymupdf.open(
                stream=data,
                filetype="pdf"
            )

            ocr_texts = []

            try:
                if (
                    document.needs_pass
                    and not document.authenticate(password or "")
                ):
                    raise ValueError(
                        "This PDF needs the correct password."
                    )

                for page in document:
                    pixmap = page.get_pixmap(
                        dpi=250,
                        colorspace=pymupdf.csRGB,
                        alpha=False
                    )

                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples
                    )

                    try:
                        ocr_texts.append(
                            pytesseract.image_to_string(
                                image,
                                lang="eng",
                                config="--oem 3 --psm 6"
                            )
                        )
                    finally:
                        image.close()

            finally:
                document.close()

        except ValueError:
            raise
        except Exception as error:
            raise ValueError(
                f"PDF OCR failed: {error}. Check your Tesseract installation."
            ) from error

        ocr_df = parse_statement_text("\n".join(ocr_texts))
        ocr_records = (
            ocr_df.to_dict("records")
            if not ocr_df.empty else []
        )

        return ocr_records, "\n".join(ocr_texts)

    # Use whichever method recognized the most rows.
    if len(native_records) > len(table_records):
        return native_records, "\n".join(page_texts)

    return table_records, "\n".join(page_texts)


def preview_pdf_text(uploaded_file, password=None, max_pages=2):
    """Optional preview helper if you kept the debug button in app.py."""
    data = uploaded_file.getvalue()

    import pdfplumber

    with pdfplumber.open(
        io.BytesIO(data),
        password=password or ""
    ) as pdf:
        native = [
            page.extract_text() or ""
            for page in pdf.pages[:max_pages]
        ]

    if any(len(text.strip()) >= 80 for text in native):
        return "\n\n".join(native)

    try:
        import pymupdf
        from PIL import Image

        pytesseract = _tesseract_setup()
        document = pymupdf.open(stream=data, filetype="pdf")
        previews = []

        try:
            if (
                document.needs_pass
                and not document.authenticate(password or "")
            ):
                raise ValueError("Incorrect PDF password.")

            for page in list(document)[:max_pages]:
                pixmap = page.get_pixmap(
                    dpi=250,
                    colorspace=pymupdf.csRGB,
                    alpha=False
                )

                image = Image.frombytes(
                    "RGB",
                    (pixmap.width, pixmap.height),
                    pixmap.samples
                )

                try:
                    previews.append(
                        pytesseract.image_to_string(
                            image,
                            lang="eng",
                            config="--oem 3 --psm 6"
                        )
                    )
                finally:
                    image.close()

        finally:
            document.close()

        return "\n\n".join(previews)

    except Exception:
        return "\n\n".join(native)


def prepare_transactions(transactions):
    if isinstance(transactions, pd.DataFrame):
        df = transactions.copy()
    else:
        df = pd.DataFrame(transactions)

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[OUTPUT_COLUMNS]

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in ["debit", "credit", "amount", "balance"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Unknown directions must not be included in spending or income totals.
    df.loc[
        df["direction"] == "Unknown", ["debit", "credit"]
    ] = None

    df["review_reason"] = df["review_reason"].fillna("").astype(str)

    # Do not silently delete identical-looking payments.
    return df.sort_values(
        "date", na_position="last"
    ).reset_index(drop=True)


def build_category_summary(transactions):
    if transactions.empty:
        return pd.DataFrame(columns=["category", "total_spent"])

    df = transactions.copy()
    df["debit"] = pd.to_numeric(
        df["debit"], errors="coerce"
    ).fillna(0)

    return (
        df.groupby("category", dropna=False)["debit"]
        .sum()
        .reset_index(name="total_spent")
        .sort_values("total_spent", ascending=False)
        .reset_index(drop=True)
    )


def build_expense_summary(transactions):
    columns = [
        "period", "total_spent", "total_received", "transaction_count"
    ]

    if transactions.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    df = transactions.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df[df["date"].notna()].copy()

    if df.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)

    df["_spent"] = pd.to_numeric(
        df["debit"], errors="coerce"
    ).fillna(0)

    df["_received"] = pd.to_numeric(
        df["credit"], errors="coerce"
    ).fillna(0)

    df["week"] = df["date"].dt.to_period("W").astype(str)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    def summary(period_column):
        return (
            df.groupby(period_column)
            .agg(
                total_spent=("_spent", "sum"),
                total_received=("_received", "sum"),
                transaction_count=("date", "count")
            )
            .reset_index()
            .rename(columns={period_column: "period"})
        )

    return summary("week"), summary("month")


def detect_suspicious_transactions(transactions):
    if transactions.empty:
        return transactions.copy()

    df = transactions.copy()
    reasons = df["review_reason"].fillna("").astype(str).tolist()
    amounts = pd.to_numeric(df["amount"], errors="coerce")
    valid_amounts = amounts[amounts > 0]

    # A large amount is a reason for review, not proof of fraud.
    if len(valid_amounts) >= 5:
        threshold = max(float(valid_amounts.median()) * 5, 20000.0)

        for index, amount in enumerate(amounts):
            if pd.notna(amount) and amount >= threshold:
                note = (
                    f"Unusually large amount for this statement "
                    f"(≥ {threshold:,.2f}); review manually"
                )
                reasons[index] = (
                    reasons[index] + "; " if reasons[index] else ""
                ) + note

    duplicates = df.duplicated(
        subset=["date", "amount", "description"],
        keep=False
    )

    for index, is_duplicate in enumerate(duplicates):
        if is_duplicate:
            reasons[index] = (
                reasons[index] + "; " if reasons[index] else ""
            ) + "Possible duplicate; verify against original statement"

    df["review_reason"] = reasons

    return df[
        df["review_reason"].str.strip().ne("")
    ].copy().reset_index(drop=True)


def analyze_statement(uploaded_file, password=None):
    """Main entry point used by the Streamlit app."""
    filename = str(
        getattr(uploaded_file, "name", "statement.pdf")
    )
    extension = os.path.splitext(filename)[1].lower()
    raw_text = ""

    if extension in {".csv", ".txt", ".xlsx", ".xls"}:
        table = _read_table(uploaded_file, extension)

        raw_text = " ".join(
            [str(value) for value in list(table.columns)]
            + table.head(10).astype(str).values.flatten().tolist()
        )

        records = _column_records(
            table, extension.upper().lstrip(".")
        )

        if not records and extension in {".txt", ".csv"}:
            raw = uploaded_file.getvalue().decode(
                "utf-8", errors="ignore"
            )
            records = parse_statement_text(raw).to_dict("records")

    elif extension == ".pdf" or not extension:
        records, raw_text = _pdf_content(
            uploaded_file.getvalue(), password
        )

    elif extension in {
        ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"
    }:
        raw_text = _ocr_image_bytes(uploaded_file.getvalue())
        records = parse_statement_text(raw_text).to_dict("records")

    else:
        raise ValueError(
            "Unsupported file. Upload PDF, CSV, TXT, Excel or a statement image."
        )

    provider = detect_statement_type(raw_text)
    transactions = prepare_transactions(records)

    if transactions.empty:
        raise ValueError(
            "No transactions were recognized. Try a CSV/Excel export "
            "if available, or use the extracted-text preview to check OCR. "
            "This layout may need another parsing rule."
        )

    transactions["source_provider"] = provider

    weekly, monthly = build_expense_summary(transactions)
    flagged = detect_suspicious_transactions(transactions)

    return {
        "transactions": transactions,
        "category_summary": build_category_summary(transactions),
        "weekly_summary": weekly,
        "monthly_summary": monthly,
        "flagged_transactions": flagged,
        "provider": provider,
        "transaction_count": len(transactions),
        "unclassified_count": int(
            (transactions["direction"] == "Unknown").sum()
        )
    }
