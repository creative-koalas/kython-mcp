"""Output masking for execution-bound credentials, before results are retained."""

from __future__ import annotations

import json


class SecretRedactor:
    """Keep incomplete secret prefixes private across output chunks."""

    def __init__(self, secrets: tuple[str, ...]) -> None:
        variants = set()
        for secret in secrets:
            if secret:
                variants.update((secret, repr(secret)[1:-1], json.dumps(secret)[1:-1]))
        self._secrets = tuple(sorted(variants, key=len, reverse=True))
        self._by_first: dict[str, tuple[str, ...]] = {}
        for secret in self._secrets:
            self._by_first[secret[0]] = (*self._by_first.get(secret[0], ()), secret)
        self._max_length = max((len(secret) for secret in self._secrets), default=0)
        self._pending = ""

    def feed(self, text: str, *, final: bool = False) -> str:
        if not self._secrets:
            return text
        value = self._pending + text
        self._pending = ""
        output = []
        index = 0
        while index < len(value):
            candidates = self._by_first.get(value[index], ())
            remaining = value[index:] if candidates and len(value) - index < self._max_length else ""
            if remaining and not final and any(
                len(secret) > len(remaining) and secret.startswith(remaining)
                for secret in candidates
            ):
                self._pending = remaining
                break
            matched = next(
                (secret for secret in candidates if value.startswith(secret, index)), None
            )
            if matched is not None:
                output.append("[REDACTED]")
                index += len(matched)
                continue
            if remaining and any(secret.startswith(remaining) for secret in candidates):
                if final:
                    output.append("[REDACTED]")
                if not final:
                    self._pending = remaining
                break
            output.append(value[index])
            index += 1
        return "".join(output)
