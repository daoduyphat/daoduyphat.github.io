//Dùng để quy để tính tổng mảng
// int tinhTongDeQuy(int arr[], int n) {
#include <stdio.h>
#include <ctype.h>
#include <stdbool.h>
#include <stdlib.h>

// Hàm tính tổng các số trong chuỗi
int sum_of_numbers(const char *str) {
    int sum = 0;
    int num = 0;
    bool is_number = false;

    for (int i = 0; str[i] != '\0'; i++) {
        if (isdigit(str[i])) {
            num = num * 10 + (str[i] - '0'); // Xây dựng số nguyên từ nhiều chữ số
            is_number = true;
        } else if (is_number) {
            sum += num;  // Gặp ký tự không phải số -> kết thúc 1 số
            num = 0;
            is_number = false;
        }
    }

    if (is_number) { // Xử lý nếu số kết thúc ở cuối chuỗi
        sum += num;
    }

    return sum;
}

int main() {
    system("cls");
    char str[] = "h13n33!a123z9";
    int total = sum_of_numbers(str);
    printf("Tong cac so trong chuoi la: %d\n", total);
    return 0;
}
