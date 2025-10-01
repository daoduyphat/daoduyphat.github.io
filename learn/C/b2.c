/*
    Chương trình C tính tổng các số nguyên theo 4 cách khác nhau.
    👉 Chỉ **CÁCH 1** (tính tổng từ 1 đến N) đang được chạy.
    👉 Các cách khác (dùng con trỏ, hàm, đệ quy) được đặt trong comment để dễ so sánh và kích hoạt khi cần.

    ✅ Cách 1: Tính tổng từ 1 đến N
    ✅ Cách 2: Dùng con trỏ duyệt mảng
    ✅ Cách 3: Dùng hàm tính tổng mảng
    ✅ Cách 4: Dùng đệ quy tính tổng mảng
*/

#include <stdio.h>
#include <stdlib.h>

// ===== CÁCH 1: Tính tổng từ 1 đến N =====
int calculateSum(int N) {
    int sum = 0;
    for (int i = 1; i <= N; i++) {
        sum += i;
    }
    return sum;
}

int main() {
    system("cls"); 
    int N;
    printf("Nhap so N: ");
    scanf("%d", &N);

    int result = calculateSum(N);
    printf("TTong tu 1 den %d la: %d\n", N, result);
    return 0;
}

/* 
// ===== CÁCH 2: Dùng con trỏ để tính tổng mảng =====
int main() {
    int arr[6] = {1, 2, 3, 4, 5, 6};
    int *p = arr;
    int sum = 0;

    for (int i = 0; i < 6; ++i) {
        sum += *p;
        p++;
    }

    printf("Tổng các phần tử trong mảng (Cách 2 - con trỏ): %d\n", sum);
    return 0;
}
*/

/*
// ===== CÁCH 3: Dùng hàm tính tổng mảng =====
int tinhTong(int arr[], int n) {
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    int arr[6] = {1, 2, 3, 4, 5, 6};
    int sum = tinhTong(arr, 6);
    printf("Tổng các phần tử trong mảng (Cách 3 - hàm thường): %d\n", sum);
    return 0;
}
*/

/*
// ===== CÁCH 4: Dùng đệ quy =====
int tinhTongDeQuy(int arr[], int n) {
    if (n == 0) return 0;
    return arr[n - 1] + tinhTongDeQuy(arr, n - 1);
}

int main() {
    int arr[6] = {1, 2, 3, 4, 5, 6};
    int sum = tinhTongDeQuy(arr, 6);
    printf("Tổng các phần tử trong mảng (Cách 4 - đệ quy): %d\n", sum);
    return 0;
}
*/
