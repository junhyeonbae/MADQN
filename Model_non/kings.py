import numpy as np


def king_adj2(n) :

    A = np.zeros((n**2, n**2))

    for i in range(n**2):

        if i // n == 0  : #첫번째 행

            if i % n  == 0 : #첫번째 열 : 몫이 0 이어야 함

                A[i, i+1] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

            elif i % n == n-1 : #마지막 열  몫이 n-1 이어야 함

                A[i, i-1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1

            else:
                A[i, i-1] = 1
                A[i, i+1] = 1
                A[i, i-1+n] = 1
                A[i, i+n] = 1
                A[i, i+1+n] = 1

        elif i // n == n-1 : #마지막 행

            if i % n ==  0 : #첫번째 열

                A[i, i-n] = 1
                A[i, i+1-n] = 1
                A[i, i+1] = 1
            elif i % n  == n-1 : #마지막 열

                A[i, i-n] = 1
                A[i, i-1-n] = 1
                A[i, i-1] = 1
            else: #중간열
                A[i, i - 1] = 1
                A[i, i + 1] = 1
                A[i, i - 1-n] = 1
                A[i, i-n] = 1
                A[i, i+ 1-n] = 1

        else:  #중간 행에 포함되는 것



                if i % n == 0 : #첫번째 열
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i+1] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1

                elif i % n == n-1 : #마지막 열
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i-1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                else:
                    A[i, i-1-n] = 1
                    A[i, i-n] = 1
                    A[i, i+1-n] = 1
                    A[i, i-1] = 1
                    A[i, i+1] = 1
                    A[i, i-1+n] = 1
                    A[i, i+n] = 1
                    A[i, i+1+n] = 1


    for i in range(n**2):
        A[i,i] = 1

    return A

print(2%30)


