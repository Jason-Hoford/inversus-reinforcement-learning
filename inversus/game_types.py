"""Type definitions for the INVERSUS game."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TileColor(Enum):
    """Tile color enumeration."""
    BLACK = "black"
    WHITE = "white"


class Direction(Enum):
    """Direction enumeration for movement and shooting."""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class ActionType(Enum):
    """Action type enumeration."""
    MOVE = "move"
    SHOOT = "shoot"
    CHARGE_SHOOT = "charge_shoot"
    NONE = "none"


class PlayerId(Enum):
    """Player identifier enumeration."""
    P1 = 1
    P2 = 2


@dataclass
class Action:
    """Represents a player action."""
    type: ActionType
    direction: Optional[Direction] = None
    
    def __post_init__(self) -> None:
        """Validate action consistency."""
        if self.type in (ActionType.MOVE, ActionType.SHOOT, ActionType.CHARGE_SHOOT):
            if self.direction is None:
                raise ValueError(f"Action type {self.type} requires a direction")
        elif self.type == ActionType.NONE:
            if self.direction is not None:
                raise ValueError("Action type NONE must have direction=None")


@dataclass
class PlayerState:
    """Represents a player's state in the game."""
    player_id: PlayerId
    x: int
    y: int
    color: TileColor  # this player's "color" (the one they CANNOT walk on)
    ammo: int  # current ammo
    max_ammo: int  # max ammo (6)
    reload_counter: int  # accumulator for reload timing
    alive: bool = True


@dataclass
class Bullet:
    """Represents a bullet in the game."""
    x: int
    y: int
    dir: Direction
    owner: PlayerId

