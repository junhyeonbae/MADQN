import numpy as np
from scipy.spatial.distance import pdist, squareform



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



# 샘플 데이터: 각 행이 한 점의 좌표를 나타냅니다.
points = np.array([[0, 1], [1, 0], [2, 2] ,[3,3]])

# pdist로 각 점 쌍 사이의 거리 계산 후, squareform으로 정사각 거리 행렬 형태로 변환
distance_matrix = squareform(pdist(points, 'euclidean'))

print(pdist(points, 'euclidean'))
print("#"*100)
print(distance_matrix)
print("#"*100)
print(distance_matrix[0:2,2:4])
print("#"*100)
print(np.mean(distance_matrix[0:2,2:4]))
print("#"*100)

import numpy as np

# 주어진 두 배열
array1 = np.array([[4, 4], [3, 17], [17, 11]], dtype=np.int32)
array2 = np.array([[8, 9], [1, 1], [8, 1]], dtype=np.int32)

# 두 배열을 세로로(수직으로) 붙이기
result_array = np.vstack((array1, array2))

distance_matrix = squareform(pdist(result_array, 'euclidean'))

print(distance_matrix)


# # 1로 채워진 (3, 3, 2) 배열 생성

# array1 = np.ones((3, 3,2))
#
# # 0으로 채워진 (3, 3, 2) 배열 생성
# array2 = np.zeros((3, 3,2))
#
# # 두 배열을 가로로 연결
# result = np.concatenate((array1, array2), axis=0)
#
# # 결과를 일렬로 펴기
# flattened_result = result.flatten()
#
# # 일렬로 펴진 결과 출력
# print(flattened_result)

matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16],
                   [17, 18, 19, 20],
                   [21, 22, 23, 24]])

print(matrix[0,1:])
print("#"*100)

import numpy as np

# 초기 배열
original_array = np.array([[3, 4],
                           [3, 17],
                           [17, 12],
                           [8, 9],
                           [1, 1],
                           [8, 1]], dtype=np.int32)

# 추가할 행
new_row = np.array([[1, 1]], dtype=np.int32)

# 배열에 새로운 행 추가
result_array = np.append(original_array, new_row, axis=0)

print(result_array)


import numpy as np

# 초기 배열
array1 = np.array([5., 3.60555128, 6.40312424])
array2 = np.array([1, 2, 3])

# 딕셔너리 초기화
summation_team_dist = {0: [], 1: []}

# 첫 번째 키의 값에 array1 할당
summation_team_dist[0] = array1

# 첫 번째 키의 값에 array2를 연결하여 다시 할당
# summation_team_dist[0] = np.concatenate((summation_team_dist[0], array2))




import wandb
import numpy as np

# WandB 세션 시작
wandb.init(project='wandb test3', entity='junhyeon')
wandb.run.name = 'wandb_test2'

import wandb
import matplotlib.pyplot as plt
# WandB 세션 시작


# 데이터 준비
x = [i for i in range(50)]  # X 데이터
y = [i ** 2 for i in range(50)]  # Y 데이터

a = [i for i in range(50)]  # A 데이터
b = [-i ** 1.5 +2 for i in range(50)]  # B 데이터

# Matplotlib를 사용하여 scatter plot 생성
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='XY')  # X, Y 데이터를 파란색으로 표시
plt.scatter(a, b, color='red', label='AB')  # A, B 데이터를 빨간색으로 표시
plt.title('XY and AB Scatter Plot')
plt.xlabel('X/A')
plt.ylabel('Y/B')
plt.legend()

# WandB에 plot 이미지 로깅
wandb.log({"XY_AB_Scatter": wandb.Image(plt)})
plt.close()  # 현재 그림 닫기
