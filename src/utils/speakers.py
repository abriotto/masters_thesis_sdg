"""Speaker normalisation for the Lai et al. (2023) ONUW corpus.

Single source of truth, shared by the transcript pipeline, the strategy
annotation analysis and the vote choice models. It exists because speaker names
are spelled inconsistently across three places -- the raw transcripts, Lai's
annotation JSONs, and the ``playerNames`` roster in the outcome files -- and an
unmatched name is not an error anywhere: the turn is simply dropped. Left
unhandled that silently costs a real player their turns, so every drop must be
either corrected here or deliberate and named here.

Three categories:

``ANNOUNCER_ALIASES``
    Game audio, timers and generic ``Speaker N`` labels. Real speech, but not a
    player; the transcript pipeline keeps these as a single ``Announcer``.

``NON_PLAYER_SPEAKERS``
    Group utterances ("All", "Everyone") and unnameable speakers. Dropped,
    deliberately.

    Only labels that can never be a person's name belong here. A personal name
    is roster-dependent: "Matt" is an off-roster bystander in one game and a
    real player in another, so listing him globally silently deleted 37 of his
    turns. Pass ``player_names`` to :func:`normalize_speaker` and the roster
    always wins; an off-roster personal name is then reported as unmatched,
    which is the honest classification.

``SPEAKER_CORRECTIONS``
    Misspellings of a name that *is* on the roster. Each one is verified against
    that game's roster; the comment records it.

Deliberately absent: ``Jordan``. That roster holds both ``Jordan1`` and
``Jordan2``, so the speaker cannot be attributed to either and the turns stay
unresolved rather than being guessed at.
"""

ANNOUNCER_ALIASES = {
    "announcer",
    "audio",
    "automated",
    "cell phone voice",
    "game audio",
    "siri",
    "siri voice",
    "timer",
    "twitch alert",
    "voiceover",
}

NON_PLAYER_SPEAKERS = {
    "all",
    "everyone",
    "friend",
    "kaelan and daniel",
    "kaelan and danieal",
    "new speaker",
    "unknown",
}

SPEAKER_CORRECTIONS = {
    "Chirs": "Chris",        # roster: Elliot, Ashley, James, Chris, Sukeshi, Sian
    "Danieal": "Daniel",     # roster: Kevin, Kaelan, Jessica, Daniel
    "Jus": "Justin",         # roster: Justin, Mike, James, Mitchell
    "Mitch": "Mitchell",     # roster: Justin, Paul, Mitchell, Mike
    "Mithcell": "Mitchell",  # roster: Justin, Dan, Mitchell, Caitlynn, Mike
}

ANNOUNCER_LABEL = "Announcer"


def is_announcer(speaker) -> bool:
    """True for game audio, timers and generic 'Speaker N' labels."""
    text = str(speaker or "").strip().lower()
    return text in ANNOUNCER_ALIASES or text.startswith("speaker ")


def is_non_player_speaker(speaker) -> bool:
    """True for anything that is not one of the game's players."""
    text = str(speaker or "").strip().lower()
    return not text or is_announcer(text) or text in NON_PLAYER_SPEAKERS


def correct_speaker(speaker) -> str:
    """Roster spelling of a speaker name, unchanged if no correction applies."""
    return SPEAKER_CORRECTIONS.get(str(speaker or "").strip(), str(speaker or "").strip())


def normalize_speaker(speaker, player_names=None):
    """Roster spelling of a player, or None if the speaker is not a player.

    ``player_names`` is the game's roster. When given it takes precedence: a
    name on the roster is a player even if it also appears in the non-player
    labels. Pass it whenever it is available.
    """
    corrected = correct_speaker(speaker)

    if player_names and corrected in player_names:
        return corrected

    if is_non_player_speaker(corrected):
        return None

    return corrected


def normalize_transcript_speaker(speaker) -> str:
    """Transcript-pipeline variant: announcer-like speakers collapse to one label.

    Group and off-roster speakers keep their name here, because the caller drops
    any speaker that is not on the roster and reports what it dropped.
    """
    if is_announcer(speaker):
        return ANNOUNCER_LABEL
    return correct_speaker(speaker)
