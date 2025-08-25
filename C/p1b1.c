#include <stdio.h>
#include <stdlib.h>

double averageEven(int arr[], int n) {
    int sum = 0, count = 0;
    for(int i = 0; i < n; i++) {
        if(arr[i] % 2 == 0) {
            sum += arr[i];
            count++;
        }
    }
    if(count == 0) return 0.0;
    return (double)sum / count;
}

int main() {
    system("cls");
    int n;
    printf("Nhap so luong phan tu: ");
    scanf("%d", &n);

    int arr[n];
    for(int i = 0; i < n; i++) {
        printf("Nhap phan tu thu %d: ", i+1);
        scanf("%d", &arr[i]);
    }

    double avg = averageEven(arr, n);
    printf("Trung binh cong cac so chan la: %.2lf\n", avg);

    return 0;
}