#include <iostream>
#include "/usr/local/elemental/0.85/PureRelease_python/include/El.hpp"

using namespace std;
using namespace El;


/*
C Wrapper for HermitianEig() in Elemental Library

Reference:
    http://libelemental.org/documentation/0.85/lapack_like/spectral/HermitianEig.html
*/

// void HermitianEig_cython(double* A, double* w, double* Z, int n){
//     /* A is passed from Python Matrix as a C array
//      need to convert it to an Elemental Matrix instance */
//     Matrix<double> A_elmatrix(n, n);

//     int i, j;
//     int index = 0;

//     for(j=0; j<n; j++){
//         for(i=0; i<n; i++){
//             A_elmatrix.Set(i, j, A[index]);
//             index++;
//         }
//     }

//     UpperOrLower uplo = UPPER;
//     Matrix<double> w_elmatrix(n, 1);
//     Matrix<double> Z_elmatrix(n, n);


//     HermitianEig(uplo, A_elmatrix, w_elmatrix, Z_elmatrix);

//     // Convert Elemental Matrix to C array
//     index = 0;
//     for(j=0; j<n; j++){
//        for(i=0; i<n; i++){
//             Z[index] = Z_elmatrix.Get(i, j);
//             index++;
//        }
//     }

//     index = 0;
//     for(i=0; i<n; i++){
//         w[index] = w_elmatrix.Get(i, 1);
//         index++;
//     }
// }


main(){
    // double in_array[3][3] = {{1,2,3}, {2,4,5}, {3,5,6}};

    int m=10, n=10;
    Matrix<double> A(m, n);

    for (int j=0: j<n; ++j){
        for(int i=0; i<m; ++i){
            A.Set(i, j, double(i-j));
        }
    }

    cout << "Print matrix: " << A.LDim();
    // for(int i=0; i<9; i++){
    //     cout << in_array[i];
    // }
}