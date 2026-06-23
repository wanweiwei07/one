"""Antipodal grasp planning for the Linkerbot O6 dexterous hand's TRIPOD.

The tripod is a thumb-vs-(index + middle) precision grasp -- still a 2-jaw
opposition as far as ``antipodal`` is concerned (the thumb is one jaw, the index
+ middle pads together form the other). It reuses the same machinery as the
pinch: ``hand.as_jaw('tripod')`` calibrates and presents the hand as a
parallel jaw, the middle finger flexed deeper so it genuinely opposes the thumb
(see o6._GRASP_TABLE). ring + pinky stay tucked.

Keys: N = next grasp as the solid one, R = re-plan.
"""
from test_o6_pinch import main


if __name__ == '__main__':
    main('tripod')
