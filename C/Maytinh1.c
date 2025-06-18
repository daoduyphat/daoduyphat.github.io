// Máy tính khi nhập số từ bàn phím.
#include <stdio.h>      // Thư viện cho các hàm nhập xuất chuẩn
#include <stdlib.h>     // Thư viện cho các hàm hệ thống như system()

int main(void) {
    system("cls");      // Xóa màn hình console (chỉ hoạt động trên Windows)
    char op;            // Biến lưu ký tự phép toán (+, -, *, /)
    double first, second; // Hai biến lưu hai số thực nhập vào
    printf("Nhap phep toan (+, -, *, /): "); // Yêu cầu người dùng nhập phép toán
    scanf(" %c", &op);                       // Đọc ký tự phép toán từ bàn phím
    printf("Nhap hai so: ");                 // Yêu cầu người dùng nhập hai số
    scanf("%lf %lf", &first, &second);       // Đọc hai số thực từ bàn phím
    switch (op) {                            // Kiểm tra phép toán vừa nhập
        case '+':
            printf("%.1lf + %.1lf = %.1lf", first, second, first + second); // Cộng hai số
            break;
        case '-':
            printf("%.1lf - %.1lf = %.1lf", first, second, first - second); // Trừ hai số
            break;
        case '*':
            printf("%.1lf * %.1lf = %.1lf", first, second, first * second); // Nhân hai số
            break;
        case '/':
            printf("%.1lf / %.1lf = %.1lf", first, second, first / second); // Chia hai số
            break;
        default:
            printf("Phep toan khong hop le"); // Thông báo nếu phép toán không hợp lệ
    }
    return 0; // Kết thúc chương trình
}
