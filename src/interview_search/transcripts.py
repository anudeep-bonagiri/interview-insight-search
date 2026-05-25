"""Load interview transcripts and split them into retrieval chunks.

An interview is a list of speaker turns, each with a timestamp. We chunk by
grouping consecutive turns up to a target word count, keeping a small overlap so
a participant's answer is never split away from the question that prompted it.
Each chunk records the turn range it covers, which is what lets the evaluation
harness map turn-level relevance labels onto retrieved chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Turn:
    idx: int
    speaker: str
    ts: str
    text: str


@dataclass
class Interview:
    id: str
    title: str
    product: str
    date: str
    interviewer: str
    participant: dict
    turns: list[Turn] = field(default_factory=list)

    @property
    def participant_name(self) -> str:
        return self.participant.get("name", "Participant")

    @property
    def participant_role(self) -> str:
        return self.participant.get("role", "")


@dataclass
class Chunk:
    chunk_id: str
    interview_id: str
    interview_title: str
    participant_name: str
    participant_role: str
    date: str
    speakers: list[str]
    start_ts: str
    end_ts: str
    start_turn: int
    end_turn: int
    text: str

    def citation(self) -> str:
        """A compact, human-readable source reference."""
        return (
            f"[{self.interview_id} {self.start_ts}-{self.end_ts}, "
            f"{self.participant_name}, {self.participant_role}]"
        )

    def covered_turns(self) -> range:
        return range(self.start_turn, self.end_turn + 1)


def load_interview(path: Path) -> Interview:
    data = json.loads(Path(path).read_text())
    turns = [
        Turn(idx=i, speaker=t["speaker"], ts=t["ts"], text=t["text"])
        for i, t in enumerate(data["turns"])
    ]
    return Interview(
        id=data["id"],
        title=data["title"],
        product=data.get("product", ""),
        date=data.get("date", ""),
        interviewer=data.get("interviewer", ""),
        participant=data.get("participant", {}),
        turns=turns,
    )


def load_interviews(data_dir: Path | str | None = None) -> list[Interview]:
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    paths = sorted(data_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No interview JSON files found in {data_dir}")
    return [load_interview(p) for p in paths]


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_interview(
    interview: Interview,
    target_words: int = config.TARGET_WORDS,
    overlap_turns: int = config.OVERLAP_TURNS,
) -> list[Chunk]:
    """Group consecutive turns into ~target_words chunks with turn overlap."""
    chunks: list[Chunk] = []
    turns = interview.turns
    i = 0
    n = 0
    while i < len(turns):
        group: list[Turn] = []
        words = 0
        j = i
        while j < len(turns):
            group.append(turns[j])
            words += _word_count(turns[j].text)
            j += 1
            if words >= target_words:
                break
        chunks.append(_make_chunk(interview, group, n))
        n += 1
        if j >= len(turns):
            break
        # Step forward, leaving `overlap_turns` of context shared with the next chunk.
        i = max(j - overlap_turns, i + 1)
    return chunks


def _make_chunk(interview: Interview, group: list[Turn], n: int) -> Chunk:
    text = "\n".join(f"{t.speaker}: {t.text}" for t in group)
    speakers: list[str] = []
    for t in group:
        if t.speaker not in speakers:
            speakers.append(t.speaker)
    return Chunk(
        chunk_id=f"{interview.id}::c{n:03d}",
        interview_id=interview.id,
        interview_title=interview.title,
        participant_name=interview.participant_name,
        participant_role=interview.participant_role,
        date=interview.date,
        speakers=speakers,
        start_ts=group[0].ts,
        end_ts=group[-1].ts,
        start_turn=group[0].idx,
        end_turn=group[-1].idx,
        text=text,
    )


def build_chunks(
    data_dir: Path | str | None = None,
    target_words: int = config.TARGET_WORDS,
    overlap_turns: int = config.OVERLAP_TURNS,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for interview in load_interviews(data_dir):
        chunks.extend(chunk_interview(interview, target_words, overlap_turns))
    return chunks
