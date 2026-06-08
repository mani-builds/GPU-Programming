#include <iostream>
#include <typeinfo>
#include <vector>
#include <cstdio>

using namespace std;

int c = 0;
void f() {
    // Static variable
    static int count = 0;
    count++;
    c++;
    // cout << count << " " << endl;
  	cout << c << " ";
}

int main() {
  	// Calling function f() 5 times
    for (int i = 0; i < 5; i++)
        f();

    int n;
    cout<<"Enter n: ";
    cin>>n;
    cout<<"n is: "<< n <<endl;
    cout<<"typeinfo: "<< std::typeof(n) <<endl;

    return 0;
}
