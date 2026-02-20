#include "MeshProcessor.h"

#include "Utils.h"

using namespace ShapeAnalysis;
MeshProcessor::MeshProcessor(const std::string& fileName)
{
    std::tie(mesh, geometry) =  surface::readSurfaceMesh(fileName);
    meshName = std::filesystem::path(fileName).stem().string();
}

MeshProcessor::~MeshProcessor()
{}

MeshProcessorSptr MeshProcessor::create(const std::string& fileName)
{
    MeshProcessorSptr obj = std::shared_ptr<MeshProcessor>(new MeshProcessor(fileName));
    return obj;
}

void MeshProcessor::computeFaceNormal()
{
    geometry->requireFaceNormals();
    faceNormals.clear();
    faceNormals = surface::FaceData<Vector3>(*mesh);
    for(const surface::Face f : mesh->faces())
    {
        const Vector3 normal =  geometry->faceNormal(f);
        faceNormals[f] = normal;
    }
    geometry->unrequireFaceNormals();
}

void MeshProcessor::computeFaceArea()
{
    geometry->requireFaceAreas();
    faceArea.clear();
    faceArea = surface::FaceData<double>(*mesh);
    for(const surface::Face f : mesh->faces())
    {
        //std::cout << "Face " << f << std::endl;
        const double area = geometry->faceArea(f);
        faceArea[f] = area;
    }
    geometry->unrequireFaceAreas();
}

const int& MeshProcessor::getEigenSpectrumSize() const
{
    return eigenSpectrumSize;
}

void MeshProcessor::laplacianSpectrum(const int no_of_ev, const bool intrinsic, const bool robust)
{
    std::cout << "\n";
    std::cout << "\033[32m" << "Computing Discrete Laplacian and Vertex Lumped Mass Matrix for mesh : " << "\033[0m" <<  meshName << std::endl;
    const bool robustLaplacian = (intrinsic || robust);
    const double relativeMollificationFactor = robust ? 1e-5 : 0.0;
    //std::cout << "relativeMollificationFactor " << relativeMollificationFactor << "\n";
    // L = Eigen::SparseMatrix<double>();
    // M = Eigen::SparseMatrix<double>();

    if(robustLaplacian)
    {
        std::tie(L, M) = surface::buildTuftedLaplacian(*mesh, *geometry, relativeMollificationFactor);
    }
    else
    {
        geometry->requireCotanLaplacian();
        geometry->requireVertexLumpedMassMatrix();
        L = geometry->cotanLaplacian;
        M = geometry->vertexLumpedMassMatrix;
        std::cout << "Matrix sizes: L (" << L.rows() << "," << L.cols() << "), M (" << M.rows() << "," << M.cols() << ")" << std::endl;
    }
    std::cout << "\033[32m" << "Done !! " << "\033[0m" << std::endl;
    L.makeCompressed();
    M.makeCompressed();

    // The lumped mass matrix is diagonal. We explicitly assert this property to guarantee correctness. Many matrix multiplications involving the
    // mass matrix are implemented using optimized element-wise operations, which assume a diagonal structure.
    assert(ShapeAnalysis::isDiagonal(M) && "Mass matrix should be diagonal");
    geometry->unrequireCotanLaplacian();
    geometry->unrequireVertexLumpedMassMatrix();
}

void MeshProcessor::readPreComputedEigenSpectrum(const int no_of_ev, const std::string& eigenSpecPath)
{
    std::cout << std::endl;
    std::cout << "\033[32m" << "Reading eigen spectra from file..." << "\033[0m" << std::endl;
    std::cout << "Note: It is recommended to precompute 200 eigenvalues and eigenvectors per mesh, ";
    std::cout << "so currently, 200 eigen spectra have been computed per mesh." << std::endl;

    std::filesystem::path dir = eigenSpecPath;
    std::string last_name = dir.filename().string();

    std::filesystem::path pathEval = dir / (last_name + "_evals.csv");
    std::filesystem::path pathEvec = dir / (last_name + "_evecs.csv");

    assert(ShapeAnalysis::isFileExist(pathEval) && "File should exist");
    assert(ShapeAnalysis::isFileExist(pathEvec) && "File should exist");

    // Read Eigen Values
    rapidcsv::Document doc_Eval(pathEval, rapidcsv::LabelParams(-1, -1));
    assert(doc_Eval.GetRowCount() == no_of_ev && doc_Eval.GetColumnCount() == 1);
    const std::vector<double>& eVal = doc_Eval.GetColumn<double>(0);
    eigenValues.resize(static_cast<Eigen::Index>(doc_Eval.GetRowCount()));
    for(int i=0; i<doc_Eval.GetRowCount(); i++)
    {
        eigenValues(static_cast<Eigen::Index>(i)) = eVal[i];
    }

    // Read Eigen Vectors
    rapidcsv::Document doc_Evec(pathEvec, rapidcsv::LabelParams(-1, -1));
    assert(doc_Evec.GetRowCount() ==  mesh->nVertices() && doc_Evec.GetColumnCount() == no_of_ev);

    eigenVectors = Eigen::MatrixXd(doc_Evec.GetRowCount(), no_of_ev);
    for(int i=0; i<doc_Evec.GetColumnCount(); i++)
    {
        std::vector<double> vec = doc_Evec.GetColumn<double>(i);
        eigenVectors.col(i) = Eigen::Map<Eigen::VectorXd>(vec.data(), static_cast<Eigen::Index>(doc_Evec.GetRowCount()));
    }
    std::cout << "Eigen Values Row: " << eigenValues.rows() << " " << "Column: " << eigenValues.cols() << std::endl;
    std::cout << "Eigen Vectors Row: " <<  eigenVectors.rows() << " " << "Column: " << eigenVectors.cols() << std::endl;
    std::cout << "\033[32m" << "Done !! " << "\033[0m" << std::endl;
}

void MeshProcessor::eigenSpectrum(const int no_of_ev, const std::string& eigenSpecPath)
{
    assert(L.size() != 0 && M.size() != 0 && "Eigen Spectrum shouldn't be empty");
    if(!eigenSpecPath.empty())
    {
        readPreComputedEigenSpectrum(no_of_ev, eigenSpecPath);
        return;
    }
    std::cout << std::endl;
    std::cout << "\033[32m" << "Computing Eigen Vectors and Values of discrete laplacian (Lx = (lambda)Mx) for the given mesh : "  << "\033[0m" << meshName << std::endl;
    std::cout << "\033[32m" << "Note: The current method uses an in-built geometry central method to compute the eigenvalues and eigenvectors of the given sparse matrix, ";
    std::cout << "and it is time-consuming. It will soon be replaced by a more efficient method." << "\033[0m" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Vector<double>> eVectors = smallestKEigenvectorsPositiveDefinite(L, M, no_of_ev, 75);
    //std::vector<Vector<double>> eVectors = smallestKEigenvectorsPositiveDefiniteTol(L, M, no_of_ev, 1e-8);

    const size_t num_of_vertices = eVectors[0].size();
    assert(eVectors.size() == no_of_ev);
    assert(num_of_vertices == mesh->nVertices());

    eigenVectors = Eigen::MatrixXd(num_of_vertices, no_of_ev);
    for(int col = 0; col < no_of_ev; col++)
    {
        eigenVectors.col(col) = Eigen::Map<Eigen::VectorXd>(eVectors[col].data(), static_cast<Eigen::Index>(num_of_vertices));
    }

    eigenValues = Eigen::VectorXd(no_of_ev);
    constexpr double eps = 1e-12;
    for(int i=0; i<no_of_ev; i++)
    {
        const Eigen::VectorXd& x = eigenVectors.col(i);
        const double numerator = x.dot(L * x);
        const double denominator = x.dot(M * x);
        assert(std::abs(denominator) > eps && "Division by (near) zero");
        eigenValues(i) = (numerator / denominator);
    }
    std::cout << eigenVectors.rows() << " " << eigenVectors.cols() << std::endl;
    std::cout << eigenValues.rows() << " " << eigenValues.cols() << std::endl;

    // for(int i=1; i<no_of_ev; i++)
    // {
    //     assert(eigenValues(i) >= eigenValues(i-1));
    // }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration<double>(end - start);
    std::cout << "\033[32m" << "Done!! " << "Time taken to compute: " << duration.count() << " seconds" << "\033[0m" << std::endl;
}

void MeshProcessor::process(const int no_of_ev, const bool skipNormals, const bool intrinsic, const bool robust,  const std::string& eigenSpecPath)
{
    /*
     * Process the LB spectrum and saves it. Additionnaly computes per-face normals
     * Parameters.
     * no_of_ev ---> Number of eigenvectors to compute
     * skip normal ---> If set to true, skip normals computation.
     * intrinsic ---> optional, use intrinsic triangulation.
     * robust ---> use tufted laplacian to compute weak laplacian as well as vertex lumped mass matrix.
     */

    eigenSpectrumSize = no_of_ev;
    if(!skipNormals)
        computeFaceNormal();

    bool usingPreComputed = false;
    // Use the pre-computed Eigen Spectra.
    if(eigenValues.size() != 0 && eigenVectors.size() !=0)
    {
        if(no_of_ev > 0 && no_of_ev <= eigenValues.rows() && no_of_ev <= eigenVectors.cols())
        {
            const Eigen::VectorXd tempEval = eigenValues.head(no_of_ev);
            const Eigen::MatrixXd tempEvec = eigenVectors.leftCols(no_of_ev);
            eigenValues =  tempEval;
            eigenVectors = tempEvec;
            usingPreComputed = true;
            std::cout << "After compression size of \n";
            std::cout << "Eigen Values--> Rows: " << eigenValues.rows() << " " << "Columns: " << eigenValues.cols() << std::endl;
            std::cout << "Eigen Vectors --> Rows: " << eigenVectors.rows() << " " << "Columns " << eigenVectors.cols() << std::endl;
        }
    }

    laplacianSpectrum(no_of_ev, intrinsic, robust);
    if(!usingPreComputed)
        eigenSpectrum(no_of_ev, eigenSpecPath);
}

const Eigen::VectorXd MeshProcessor::getTruncatedEval(const int num) const
{
    if ( !(num > 0 && num <= eigenValues.rows() && num <= eigenVectors.cols()) )
    {
        throw std::out_of_range("Index out of range");
    }
    return eigenValues.head(num);

}

const Eigen::MatrixXd MeshProcessor::getTruncatedEvec(const int num) const
{
    if ( !(num > 0 && num <= eigenValues.rows() && num <= eigenVectors.cols()) )
    {
       throw std::out_of_range("Index out of range");
    }
    return eigenVectors.leftCols(num);
}

const Eigen::SparseMatrix<double>& MeshProcessor::getLaplacian() const
{
    return L;
}

const Eigen::SparseMatrix<double>& MeshProcessor::getMassMatrix() const
{
    return M;
}

// l2_sqnorm
Eigen::VectorXd MeshProcessor::computeBilinearForm(const Eigen::MatrixXd& func1, const Eigen::MatrixXd& func2) const
{
    if(func1.cols()==1)
    {
        /*
        Eigen::VectorXd func1_vec = func1.col(0); // (N,)

        const Eigen::MatrixXd M_func2 = this->M * func2; (N, P)

        Eigen::RowVectorXd temp = func1_vec.transpose() * M_func2;  // (1,N) * (N,P) = (1, P)
        Eigen::VectorXd result = temp.transpose();
        */
        const Eigen::VectorXd func1_vec = func1.col(0);
        Eigen::VectorXd result = (this->M * func2).transpose() * func1_vec;
        return result;
    }
    else
    {
        assert(func1.cols() == func2.cols() && "func1 and func2 must have same number of columns");
        const int num_cols = static_cast<int>(func1.cols());
        Eigen::VectorXd result(num_cols);

        Eigen::MatrixXd M_func2 = this->M * func2;

        // compute column wise dot product
        for(int i=0; i<num_cols; i++)
        {
            result(i) = func1.col(i).dot(M_func2.col(i));
        }
        return result;
    }
}

Eigen::MatrixXd MeshProcessor::project(const Eigen::MatrixXd& descr, const int K)
{
    Eigen::MatrixXd reduced_descr;
    if(K==-1)
    {
        //reduced_descr = eigenVectors.transpose() * (M * descr);
        Eigen::VectorXd voronoi_area = M.diagonal();
        reduced_descr = eigenVectors.transpose() * (descr.array().colwise() * voronoi_area.array()).matrix();
    }
    else if(K <= eigenVectors.cols())
    {
        const Eigen::MatrixXd& reduced_evecs = eigenVectors.leftCols(K);
        //reduced_descr = reduced_evecs.transpose() * (M * descr);
        Eigen::VectorXd voronoi_area = M.diagonal();
        // (n x k)^T @ (n x n) @ (n x p) ---> (k x p)
        reduced_descr.noalias() = reduced_evecs.transpose() * (descr.array().colwise() * voronoi_area.array()).matrix();
    }
    return reduced_descr;
}

void MeshProcessor::computeBarycentricBasis()
{
    if(faceNormals.size() == 0)
        computeFaceNormal();

    barycentric_basis.resize(3 * mesh->nFaces());

    for(const surface::Face& F: mesh->faces())
    {
        Vector3 vertices_positions[3];
        unsigned count = 0;
        for(const surface::Vertex& V : F.adjacentVertices())
        {
            vertices_positions[count] = geometry->vertexPositions[V];
            count++;
        }

        // const Vector3 basis1 = cross(faceNormals[F], vertices_positions[2] - vertices_positions[1]);
        // const Vector3 basis2 = cross(faceNormals[F], vertices_positions[0] - vertices_positions[2]);
        // const Vector3 basis3 = cross(faceNormals[F], vertices_positions[1] - vertices_positions[0]);
        // barycentric_basis.push_back(basis1);
        // barycentric_basis.push_back(basis2);
        // barycentric_basis.push_back(basis3);
        barycentric_basis[(F.getIndex() * 3) + 0] = cross(faceNormals[F], vertices_positions[2] - vertices_positions[1]);
        barycentric_basis[(F.getIndex() * 3) + 1] = cross(faceNormals[F], vertices_positions[0] - vertices_positions[2]);
        barycentric_basis[(F.getIndex() * 3) + 2] = cross(faceNormals[F], vertices_positions[1] - vertices_positions[0]);
    }
}

std::vector<Eigen::MatrixXd> MeshProcessor::computeGradient(const Eigen::MatrixXd& function, const bool normalize, const bool useSymmetry)
{
    if(barycentric_basis.empty())
        computeBarycentricBasis();

    if(faceArea.size() == 0)
        computeFaceArea();

    if(faceNormals.size() == 0)
        computeFaceNormal();

    std::vector<Eigen::MatrixXd> gradients;
    gradients.reserve(mesh->nFaces());
    for(const surface::Face& F : mesh->faces())
    {
        Vector3 vertices_pos[3];
        surface::Vertex vertex_index[3];
        unsigned count_pos = 0;
        unsigned count_vxt = 0;
        for(const surface::Vertex& V : F.adjacentVertices())
        {
            vertices_pos[count_pos++] = geometry->vertexPositions[V];
            vertex_index[count_vxt++] = V;
        }

        if(!useSymmetry)
        {
            const Vector3 grad2 = cross(faceNormals[F],  vertices_pos[0] - vertices_pos[2]) / (2.0 * faceArea[F]);
            const Vector3 grad3 = cross(faceNormals[F], vertices_pos[1] - vertices_pos[0]) / (2.0 * faceArea[F]);
            // if(function.cols() == 1)
            // {
            //     const Eigen::VectorXd fun = function.col(0);
            //     const double f1 = fun( vertex_index[0].getIndex());
            //     const double f2 = fun( vertex_index[1].getIndex());
            //     const double f3 = fun( vertex_index[2].getIndex());
            //     Vector3 gradient =  (f2 - f1) * grad2 + (f3 - f1) * grad3;
            // }
            {
                //function ----> (N x num_time_step)  N , num of vertices
                const Eigen::RowVectorXd f1 = function.row(vertex_index[0].getIndex());
                const Eigen::RowVectorXd f2 = function.row(vertex_index[1].getIndex());
                const Eigen::RowVectorXd f3 = function.row(vertex_index[2].getIndex());
                const Eigen::RowVectorXd df2 = f2 - f1; // (1x num_time_step)
                const Eigen::RowVectorXd df3 = f3 - f1; // (1x num_time_step)
                const Eigen::RowVector3d g2(grad2.x, grad2.y, grad2.z); // (1 x3)
                const Eigen::RowVector3d g3(grad3.x, grad3.y, grad3.z); // (1 x3)
                Eigen::MatrixXd grad(function.cols(), 3);
                grad.noalias() = (df2.transpose() * g2) + (df3.transpose() * g3);  // (num_time_step x 1) @ (1 x 3) + (num_time_step x 1) @ (1 x 3)  ---> (num_time_step x 3)
                gradients.push_back(grad);
            }
        }
        else
        {
            // const Vector3 grad1 = cross(faceNormals[F], vertices_pos[2] - vertices_pos[1]) / (2.0 * faceArea[F]);
            // const Vector3 grad2 = cross(faceNormals[F], vertices_pos[0] - vertices_pos[2]) / (2.0 * faceArea[F]);
            // const Vector3 grad3 = cross(faceNormals[F], vertices_pos[1] - vertices_pos[0]) / (2.0 * faceArea[F]);
            const double invA = 1.0 / (2.0 * faceArea[F]);
            const Vector3 grad1 = barycentric_basis[(F.getIndex() * 3) + 0] * invA;
            const Vector3 grad2 = barycentric_basis[(F.getIndex() * 3) + 1] * invA;
            const Vector3 grad3 = barycentric_basis[(F.getIndex() * 3) + 2] * invA;

            {
                const Eigen::RowVector3d g1(grad1.x, grad1.y, grad1.z); // (1 x3)
                const Eigen::RowVector3d g2(grad2.x, grad2.y, grad2.z); // (1 x3)
                const Eigen::RowVector3d g3(grad3.x, grad3.y, grad3.z); // (1 x3)

                const Eigen::RowVectorXd f1 = function.row(vertex_index[0].getIndex());
                const Eigen::RowVectorXd f2 = function.row(vertex_index[1].getIndex());
                const Eigen::RowVectorXd f3 = function.row(vertex_index[2].getIndex());
                Eigen::MatrixXd grad(function.cols(), 3);
                grad.noalias() = (f1.transpose() * g1) + (f2.transpose() * g2) + (f3.transpose() * g3);
                gradients.push_back(grad);
            }
        }
    }
    if(normalize)
    {
        for(Eigen::MatrixXd& G: gradients)
        {
            Eigen::VectorXd norms = G.rowwise().norm();
            norms = norms.array().max(1e-12); // avoid division by zero
            G.array().colwise() /= norms.array();
        }
    }
    return gradients;
}

Eigen::SparseMatrix<double> MeshProcessor::computeOrientationOperator(const std::vector<Eigen::MatrixXd>& gradF, const bool rotated)
{
    if(barycentric_basis.empty())
        computeBarycentricBasis();

    if(faceArea.size() == 0)
        computeFaceArea();

    if(faceNormals.size() == 0)
        computeFaceNormal();

    constexpr double inv3 = 1.0 / 3.0;
    constexpr double inv2 = 1.0 / 2.0;
    assert(gradF.size() == mesh->nFaces() && "The size must match");

    // We'll build a sparse matrix W where W*g computes ⟨ n × grad(f), grad(g) ⟩ or ⟨ n x grad(f), grad(g) ⟩ = n . ⟨grad(f) x grad(g)⟩
    std::vector<Eigen::Triplet<double>> T;
    T.reserve(12 * mesh->nFaces());
    for(const surface::Face& F : mesh->faces())
    {
        const int face_index = F.getIndex();

        //Gradient field is already rotated
        const Eigen::MatrixXd& g = gradF[face_index];
        assert(g.rows() == 1 && g.cols() == 3 && "Each face should have only one gradient");
        Vector3 grad { g(0,0), g(0,1), g(0,2) };

        // compute the rotated gradient: R = n x grad(f)
        // This is grad(f) rotated 90 degree counterclockwise in the tangent plane.
        Vector3 R = rotated ? grad : cross(faceNormals[F], grad);

        Vector3 vertices_pos[3];
        surface::Vertex vertices_index[3];
        unsigned count_pos = 0;
        unsigned count_vxt = 0;
        for(const surface::Vertex& V : F.adjacentVertices())
        {
            vertices_pos[count_pos++] = geometry->vertexPositions[V];
            vertices_index[count_vxt++] = V;
        }

        /*
            * compute gradient of barycentric basis functions
            * for triangle with vertices (v0, v1, v2), the gradients are:
            * ∇φ0 = (n x (v2 - v1)) / 2A but we omit /(2A) as it cancels]
            * ∇φ1 = (n x (v0 - v2)) / 2A
            * ∇φ2 = (n x (v1 - v0)) / 2A
            * Note: Area factor omitted since it appears in both numerator/denominator
        */
        // Vector3 grad_phi0 = cross(faceNormals[F], vertices_pos[2] - vertices_pos[1]) * inv2;
        // Vector3 grad_phi1 = cross(faceNormals[F], vertices_pos[0] - vertices_pos[2]) * inv2;
        // Vector3 grad_phi2 = cross(faceNormals[F], vertices_pos[1] - vertices_pos[0]) * inv2;

        const Vector3 grad_phi0 = barycentric_basis[(F.getIndex() * 3) + 0] * inv2;
        const Vector3 grad_phi1 = barycentric_basis[(F.getIndex() * 3) + 1] * inv2;
        const Vector3 grad_phi2 = barycentric_basis[(F.getIndex() * 3) + 2] * inv2;
        /*
            * compute face-local stiffness matrix entries
            * s_ij = (1/3) *  ⟨n x grad(f), ∇φ_i⟩
            * integrated over triangle Using lumped mass matrix approximation (1/3 of area to each vertex)
        */

        /*
            * Edge v0-v1 contributions:
            * S(v0,v1) = (1/3) ⟨ R, ∇φ1 ⟩  [φ1 is basis for v1]
            * S(v1,v0) = (1/3) ⟨ R, ∇φ0 ⟩  [φ0 is basis for v0]
        */
        const double s01 = dot(grad_phi1, R) * inv3;  // previously rotated_gradient[face_index] instead of R
        const double s10 = dot(grad_phi0, R) * inv3;  // previously rotated_gradient[face_index] instead of R

        /*
            * Edge v1-v2 contributions:
            * S(v1,v2) = (1/3) ⟨ R, ∇φ2 ⟩  [φ2 is basis for v2]
            * S(v2,v1) = (1/3) ⟨ R, ∇φ1 ⟩  [φ1 is basis for v1]
        */
        const double s12 = dot(grad_phi2, R) * inv3; // previously rotated_gradient[face_index] instead of R
        const double s21 = dot(grad_phi1, R) * inv3; // previously rotated_gradient[face_index] instead of R

        /*
            * Edge v2-v0 contributions:
            * S(v2,v0) = (1/3) ⟨ R, ∇φ0 ⟩  [φ0 is basis for v0]
            * S(v0,v2) = (1/3) ⟨ R, ∇φ2 ⟩  [φ2 is basis for v2]
        */
        const double s20 = dot(grad_phi0, R) * inv3; // previously rotated_gradient[face_index] instead of R
        const double s02 = dot(grad_phi2, R) * inv3; // previously rotated_gradient[face_index] instead of R

        const int v0 = vertices_index[0].getIndex();
        const int v1 = vertices_index[1].getIndex();
        const int v2 = vertices_index[2].getIndex();

        /*
            * Helper to add all 4 entries for an edge (i,j)
            * Creates antisymmetric stencil:
            * W(i,j) = s_ij
            * W(j,i) = s_ji
            * W(i,i) = -s_ij  (for conservation)
            * W(j,j) = -s_ji  (for conservation)
        */
        auto add_edge = [&](const int i, const int j, const double s_ij, const double s_ji)
        {
            T.emplace_back(i, j, s_ij);
            T.emplace_back(j, i, s_ji);   // symmetric partner
            T.emplace_back(i, i, -s_ij);
            T.emplace_back(j, j, -s_ji);
        };
        // Add contributions from all three edges of the triangle
        add_edge(v0, v1, s01, s10); // Edge between v0 and v1
        add_edge(v1, v2, s12, s21); // Edge between v1 and v2
        add_edge(v2, v0, s20, s02); // Edge between v2 and v0
    }

    // Assemble sparse matrix from triplets
    Eigen::SparseMatrix<double> W(mesh->nVertices(), mesh->nVertices());
    W.setFromTriplets(T.begin(), T.end());

    // Eigen::SparseMatrix<double> invM(mesh->nVertices(), mesh->nVertices());
    // invM.reserve(M.rows()); // reserve for diagonal entries only.
    // for(int i=0; i<M.rows(); i++)
    // {
    //     invM.insert(i, i) = 1.0 /M.coeff(i, i);
    // }
    // invM.makeCompressed();
    // Eigen::SparseMatrix<double> O = invM * W;
    // O.makeCompressed();
    // return O;
    for(int k = 0; k < W.outerSize(); ++k)
    {
        for (Eigen::SparseMatrix<double>::InnerIterator it(W, k); it; ++it)
        {
            it.valueRef() /= M.coeff(it.row(), it.row());
        }
    }
    W.makeCompressed();
    return W;
}

// Eigen vector and value computation using Spectra Library.
// std::cout << "\033[32mComputing " << no_of_ev << " eigenvalues (sigma=-0.01) " << "\033[0m" << std::endl;
// auto start = std::chrono::high_resolution_clock::now();
//
// const int nev = no_of_ev;
// const int ncv = 2 * nev; // Number of Arnoldi vectors
// const double sigma = -0.01;
// try
// {
//     // shift-and-invert for generalized problem
//     // create a shift-invert operation for (L, M)
//     Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse> op(L, M);
//
//     // mass matrix operator
//     Spectra::SparseSymMatProd<double> opM(M);
//
//     Spectra::SymGEigsShiftSolver<Spectra::SymShiftInvert<double, Eigen::Sparse, Eigen::Sparse>, Spectra::SparseSymMatProd<double>, Spectra::GEigsMode::ShiftInvert> eigs(op, opM, nev, ncv, sigma);
//     eigs.init();
//
//     int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 1000, 1e-10, Spectra::SortRule::SmallestAlge);
//     if(eigs.info() == Spectra::CompInfo::Successful)
//     {
//         eigenValues = eigs.eigenvalues();
//         eigenVectors = eigs.eigenvectors();
//     }
// }
// catch(const std::exception& e)
// {
//     std::cerr << "\033[31m" << "Error: " << e.what() << "\033[0m" << std::endl;
// }
// auto end = std::chrono::high_resolution_clock::now();
// auto duration = std::chrono::duration<double>(end - start);
// std::cout << "\033[32m" << "Time: " << duration.count() << " seconds" << "\033[0m" << std::endl;