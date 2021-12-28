import math

def pad( num, maxNum ): # returns num with 0 padding so it has same number of digits as maxNum
    nDigits_maxNum = 0
    while maxNum >= 10:
        maxNum /= 10
        nDigits_maxNum += 1
    if maxNum != 0:
        nDigits_maxNum += 1

    num_temp = num
    nDigits_num = 0
    while num_temp >= 10:
        num_temp /= 10
        nDigits_num += 1
    if num_temp != 0:
        nDigits_num += 1
    if num == 0:
        nDigits_num = 1
    
    padding = ""
    for nDigit in range( nDigits_num, nDigits_maxNum ):
        padding += "0"

    return padding + str( num )