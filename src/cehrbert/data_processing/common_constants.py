"""
Common constants extracted from cehrbert_data dependency.

These constants are used throughout the CEHR-BERT codebase for
tokenization and data processing.
"""

from datetime import datetime
from enum import Enum
from typing import Callable, Optional


# Special concept values
UNKNOWN_CONCEPT = "-1"
NA = -1  # Used for missing numeric values


class AttType(Enum):
    """Artificial Time Token (ATT) types for temporal encoding."""
    
    CEHR_BERT = "cehr_bert"  # Original CEHR-BERT variable-length logic
    DAY = "day"              # Daily granularity
    WEEK = "week"            # Weekly granularity  
    MONTH = "month"          # Monthly granularity
    MIX = "mix"              # Mixed granularity
    NONE = "none"            # No ATT tokens
    
    @property
    def value(self):
        """Return the string value of the enum."""
        return self._value_


def att_day_function(current_date: datetime, reference_date: datetime) -> Optional[int]:
    """Calculate ATT tokens based on day difference."""
    if current_date is None or reference_date is None:
        return None
    delta = current_date - reference_date
    return delta.days


def att_week_function(current_date: datetime, reference_date: datetime) -> Optional[int]:
    """Calculate ATT tokens based on week difference."""
    if current_date is None or reference_date is None:
        return None
    delta = current_date - reference_date
    return delta.days // 7


def att_month_function(current_date: datetime, reference_date: datetime) -> Optional[int]:
    """Calculate ATT tokens based on month difference (approximate)."""
    if current_date is None or reference_date is None:
        return None
    delta = current_date - reference_date
    return delta.days // 30


def att_cehr_bert_function(current_date: datetime, reference_date: datetime) -> Optional[int]:
    """
    Original CEHR-BERT ATT function with variable granularity.
    
    Uses different granularities based on time distance:
    - Days for recent events
    - Weeks for medium-term events  
    - Months for long-term events
    """
    if current_date is None or reference_date is None:
        return None
    
    delta = current_date - reference_date
    days = delta.days
    
    # Use days for first week
    if days <= 7:
        return days
    # Use weeks for first month
    elif days <= 30:
        return 7 + (days - 7) // 7
    # Use months afterwards
    else:
        return 7 + 3 + (days - 30) // 30


def att_none_function(current_date: datetime, reference_date: datetime) -> Optional[int]:
    """No ATT tokens."""
    return None


def att_mix_function(current_date: datetime, reference_date: datetime) -> Optional[int]:
    """
    Mixed granularity function (placeholder - needs specific implementation).
    
    This should be customized based on specific requirements.
    """
    # For now, use the same as cehr_bert
    return att_cehr_bert_function(current_date, reference_date)


def get_att_function(att_type: str) -> Callable[[datetime, datetime], Optional[int]]:
    """
    Get the appropriate ATT function based on the type.
    
    Args:
        att_type: String representation of AttType
        
    Returns:
        Function that calculates ATT tokens between two dates
    """
    att_function_map = {
        AttType.DAY.value: att_day_function,
        AttType.WEEK.value: att_week_function,
        AttType.MONTH.value: att_month_function,
        AttType.CEHR_BERT.value: att_cehr_bert_function,
        AttType.NONE.value: att_none_function,
        AttType.MIX.value: att_mix_function,
    }
    
    return att_function_map.get(att_type, att_none_function)