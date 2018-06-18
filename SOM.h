/*************************************************************************
* Self Organizing Maps implementation
*************************************************************************
** @file    SOM.h
** @date    26.01.2018
** @author  Yasin Yıldırım <yildirimyasi(at)gmail(dot)com>
** @copyright Copyright (c) 2018-present, Yasin Yıldırım
** @license See attached LICENSE.txt
************************************************************************/

//document this file.
/*! \file */

#pragma once
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cmath>

//include yaml-cpp library
#include <yaml-cpp/yaml.h>

//#define INTERNAL_UNIT_TEST

#define ENABLE_ROCKSDB 0

#if ENABLE_ROCKSDB
//include rocksdb
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#endif

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif // !M_PI

/// supported file formats for SOM
enum class SOMFileFormat : unsigned char
{
	YAML = 0,   ///< yaml file format
	ROCKSDB = 1 ///< RocksDB DB format
};

/// <summary>
/// Distribution types for BMU neighborhood
///  update coefficients.
/// </summary>
enum class BMDistType : unsigned char
{
	/// <summary>
	/// Same coeffs for BMU and all of its neighborhoods.
	/// </summary>
	Uniform = 0,
	/// <summary>
	/// exponential decay
	/// </summary>
	ExpDecay = 1,
	/// <summary>
	/// Gaussian distribution
	/// </summary>
	Gaussian = 2
};

/// <summary>
/// Distance metrics
/// </summary>
enum class DistanceType : unsigned char
{

	/// Euclidean distance:
	/// For 2D, the distance between \f$ (x_1,y_1) \f$ and \f$ (x_2,y_2) \f$ is
	/// \f$ \sqrt{ (x_2 - x_1)^2 + (y_2 - y_1)^2 } \f$ .
	/// General form (\f$L^2\f$ norm) for the vectors \f$A,B\f$
	/// with the size \f$n\f$ is calculated as:
	///
	/// \f[
	///   \sqrt{\sum\limits_{i=1}^n (A_i - B_i)^2}
	/// \f]
	///
	Euclidean = 0,

	/// Dot Product:
	/// General form for the vectors \f$A,B\f$  with a size \f$n\f$:
	///
	/// \f[
	///   \sum\limits_{i=1}^n (A_i * B_i)
	/// \f]
	DotProduct = 1,

	/// Cosine Simiarity:
	/// General form for the vectors \f$A,B\f$  with a size \f$n\f$ :
	///
	/// \f[
	///
	/// \text{similarity}= \cos(\theta )={\mathbf{A} \cdot \mathbf{B}
	///  \over \|\mathbf{A} \|\|\mathbf{B} \|}=
	/// {
	///    \frac {\sum \limits _{i=1}^{n}{A_{i}B_{i} } }
	///    {
	///      \sqrt {\sum \limits _{i=1}^{n}{A_{i}^{2} } }
	///      \sqrt {\sum \limits _{i=1}^{n}{B_{i}^{2} } }
	///    }
	///  }
	/// \f]
	///
	CosineSimiarity = 2,

	/// Squared Euclidean:
	/// For 2D, the distance between \f$(x_1,y_1)\f$ and \f$(x_2,y_2)\f$ is
	/// \f$ (x_2 - x_1) ^ 2 + (y_2 - y_1) ^ 2 \f$.
	/// General form for the vectors \f$A,B\f$  with a size \f$n\f$
	/// is calculated as:
	///
	/// \f[
	///   \sum\limits_{i=1}^n (A_i - B_i)^2
	/// \f]
	SquaredEuclidean = 3
};

/// <summary>
/// Self-Organizing Maps implementation.
/// </summary>
template <class T>
class SOM
{
  public:
#if defined(INTERNAL_UNIT_TEST)
	friend class SomUnitTest;
#endif

	/// <summary>
	/// Overloaded Constructor.
	/// Weights are randomly assigned between [0,1).
	/// </summary>
	/// <param name="w">width</param>
	/// <param name="h">height</param>
	/// <param name="d">#N of dimensions (codebook size).</param>
	SOM(int w, int h, int d) : W(w), H(h), D(d),
							   bmdistType(BMDistType::Uniform),
							   distanceType(DistanceType::Euclidean),
							   weights(w * h * d, static_cast<T>(0.0))
	{
		srand(time(NULL));
		for (int i = 0; i < w * h * d; ++i)
		{
			T val = static_cast<T>((rand() % 10001) / 10002.0);
			weights[i] = val;
		}
	}

	/// <summary>
	/// Overloaded Constructor.
	/// Weights are randomly assigned between [0,1).
	/// </summary>
	/// <param name="w">Width.</param>
	/// <param name="h">Height.</param>
	/// <param name="d">#N of Dimensions.</param>
	/// <param name="bmdistType"> BMU update coefficients type. </param>
	/// <param name="distanceType">Distance metric type to use. </param>
	SOM(int w, int h, int d,
		BMDistType bmdistType,
		DistanceType distanceType) : W(w), H(h), D(d),
									 bmdistType(bmdistType),
									 distanceType(distanceType),
									 weights(w * h * d, static_cast<T>(0.0))
	{
		srand(time(NULL));
		//weights.reserve(w*h*d);
		for (int i = 0; i < w * h * d; ++i)
		{
			T val = static_cast<T>((rand() % 10001) / 10002.0);
			//weights.push_back(val);
			weights[i] = val;
		}
	}
	/// <summary>
	/// trains the SOM. If there are
	/// less samples than th #N of
	/// iterations, then the samples
	///  are repeated cyclically.
	/// </summary>
	/// <param name="samples">training samples with size of N*D
	/// where N is the number of samples and
	///  D is the number of dimensions of SOM.</param>
	/// <param name="iterations">#N of iterations</param>
	/// <param name="s_learn_rate">starting learning_rate</param>
	/// <param name="f_learn_rate">ending learning_rate</param>
	/// <param name="neighborhoodSize">neighborhood size,
	///  currently only sqare neighborhood is supported.</param>
	void train(const std::vector<std::vector<T>> &samples,
			   unsigned int iterations, double s_learn_rate, double f_learn_rate,
			   double neighborhoodSize)
	{
		//tot is total number of samples.
		unsigned int tot = samples.size();
		if (s_learn_rate < f_learn_rate)
		{
			f_learn_rate = 0;
		}
		double diffLR = s_learn_rate - f_learn_rate;
		// if total number of samples (tot) is less than
		// the number of iterations, then we use cyclic
		// turn of samples.
		bool less_samples = false;
		if (tot < iterations)
			less_samples = true;
		for (unsigned int iter = 0; iter < iterations; ++iter)
		{
			diffLR *= (1.0 - iter / static_cast<double>(iterations));
			double curr_learn_rate = f_learn_rate + diffLR;
			neighborhoodSize *= (1.0 - iter / static_cast<double>(iterations));
			int x = 0, y = 0;

			// we use cyclic repeat of samples
			// if we don't have adequate samples.
			// this is to avoid index out of range error.
			int samples_idx = less_samples ? iter % samples.size() : iter;
			calcBestMatchingUnit(samples[samples_idx], y, x);
			int nSI = std::max(
				static_cast<int>(round(neighborhoodSize)), 0);
			int minX = std::max(0, x - nSI);
			int minY = std::max(0, y - nSI);
			int maxX = std::min(W - 1, x + nSI);
			int maxY = std::min(H - 1, y + nSI);

			//TODO: replace type check
			//with function objects.

			//update weights of the BMU and its neighborhoods.
			switch (bmdistType)
			{
			case BMDistType::Uniform:
			{
				for (int i = minY; i <= maxY; ++i)
				{
					for (int j = minX; j <= maxX; ++j)
					{
						T *const wi = nodeAt(i, j);
						for (int k = 0; k < D; ++k)
						{
							T error = samples[samples_idx][k] - wi[k];
							wi[k] += static_cast<T>(error * curr_learn_rate);
						}
					}
				}
				break;
			}
			case BMDistType::ExpDecay:
			{
				for (int i = minY; i <= maxY; ++i)
				{
					for (int j = minX; j <= maxX; ++j)
					{
						T *const wi = nodeAt(i, j);
						double _coef = exp((x - i) * (y - i) * (x - i) * (y - i) / (-2.0 * neighborhoodSize * neighborhoodSize));
						for (int k = 0; k < D; ++k)
						{
							T error = samples[samples_idx][k] - wi[k];
							wi[k] += static_cast<T>(error * curr_learn_rate * _coef);
						}
					}
				}
				break;
			}
			case BMDistType::Gaussian:
			{
				for (int i = minY; i <= maxY; ++i)
				{
					for (int j = minX; j <= maxX; ++j)
					{
						T *const wi = nodeAt(i, j);
						T gaussian_coef = calcGaussian2D(x, y,
														 static_cast<T>(neighborhoodSize / 2.0), j, i);
						for (int k = 0; k < D; ++k)
						{
							T error = samples[samples_idx][k] - wi[k];
							wi[k] += static_cast<T>(error * curr_learn_rate * gaussian_coef);
						}
					}
				}
				break;
			}
			default:
			{
				break;
			}
			}
		}
	}
	/// <summary>
	/// clusters the input sample.
	/// </summary>
	/// <param name="sample">input sample</param>
	/// <returns> Winner neuron's weight vector,
	///  which corresponds to the most similar
	///  weights to input pattern. </returns>
	std::vector<T> cluster(const std::vector<T> &sample)
	{
		int x, y;
		T dist = calcBestMatchingUnit(sample, y, x);
		T *const res = nodeAt(y, x);
		std::vector<T> result;
		for (int i = 0; i < D; ++i)
		{
			result.push_back(res[i]);
		}
		return result;
	}

	/// <summary>
	/// Empty destructor.
	/// </summary>
	virtual ~SOM()
	{
	}

	/// <summary>
	/// get node (neuron) weights at given position.
	/// </summary>
	/// <param name="i"> index of the first dimension (rows) of the SOM lattice.  </param>
	/// <param name="j"> index of the second dimension (columns) of the SOM lattice.</param>
	/// <returns>returns the pointer to Type T, which is the first
	/// element in the weight (codebook) vector of the corresponding SOM node.</returns>
	inline T *const nodeAt(int i, int j) const
	{
		return const_cast<T *const>(&(weights[D * (i * W + j)]));
	}

	/// <summary>
	/// assign values to weights of the neuron at given indices.
	/// </summary>
	/// <param name="i"> index at 0th dimension (rows)</param>
	/// <param name="j">index at 1th dimension (columns)</param>
	/// <param name="val">value to set.</param>
	inline void setNodeAt(int i, int j, const std::vector<T> &val)
	{
		for (int k = 0; k < D; ++k)
		{
			weights[D * (i * W + j) + k] = val[k];
		}
	}

	/// <summary>
	/// loads the trained SOM network from the file.
	/// </summary>
	/// <param name="model_path">model path</param>
	/// /// <param name="ff">file format</param>
	void load(const std::string &model_path,
			  const SOMFileFormat &ff)
	{
		switch (ff)
		{
		case SOMFileFormat::YAML:
		{
			YAML::Node model = YAML::LoadFile(model_path);
			W = model["W"].as<int>();
			H = model["H"].as<int>();
			D = model["D"].as<int>();
			distanceType = static_cast<DistanceType>(model["DistanceType"].as<unsigned char>());
			bmdistType = static_cast<BMDistType>(model["BMDistType"].as<unsigned char>());
			weights = model["weights"].as<std::vector<T>>();

			break;
		}
#if ENABLE_ROCKSDB
		case SOMFileFormat::ROCKSDB:
		{
			rocksdb::DB *db;
			rocksdb::Options options;
			// Optimize RocksDB. This is the easiest way to get RocksDB to perform well
			options.IncreaseParallelism();
			options.OptimizeLevelStyleCompaction();

			// open DB
			rocksdb::Status s = rocksdb::DB::Open(options, model_path, &db);
			assert(s.ok());
			// get value
			std::string w, h, d, bmdt, dt, strWeights;
			s = db->Get(rocksdb::ReadOptions(), "W", &w);
			W = std::stoi(w);
			assert(s.ok());

			s = db->Get(rocksdb::ReadOptions(), "H", &h);
			H = std::stoi(h);
			assert(s.ok());

			s = db->Get(rocksdb::ReadOptions(), "D", &d);
			D = std::stoi(d);
			assert(s.ok());

			s = db->Get(rocksdb::ReadOptions(), "DistanceType", &dt);
			distanceType = static_cast<DistanceType>(std::stoi(dt));
			assert(s.ok());

			s = db->Get(rocksdb::ReadOptions(), "BMDistType", &bmdt);
			bmdistType = static_cast<BMDistType>(std::stoi(bmdt));
			assert(s.ok());
			//get weights
			s = db->Get(rocksdb::ReadOptions(), "weights", &strWeights);
			assert(s.ok());
			std::stringstream ss(strWeights);
			//clear old weights, we will overwrite.
			weights.clear();
			T w_i;
			while (ss >> w_i)
			{
				weights.push_back(w_i);
				if (ss.peek() == ',')
					ss.ignore();
			}
			//close db
			delete db;
			break;
		}
#endif
		default:
			break;
		}
	}
	/// <summary>
	/// saves the trained SOM to the file.
	/// </summary>
	/// <param name="model_path">model file path</param>
	/// <param name="ff">file format</param>
	void save(const std::string &model_path,
			  const SOMFileFormat &ff)
	{

		switch (ff)
		{
		case SOMFileFormat::YAML:
		{
			std::ofstream ofile;
			ofile.open(model_path);
			YAML::Emitter out;
			//begin writing params
			out << YAML::BeginMap;
			out << YAML::Key << "W";
			out << YAML::Value << W;

			out << YAML::Key << "H";
			out << YAML::Value << H;

			out << YAML::Key << "D";
			out << YAML::Value << D;

			out << YAML::Key << "DistanceType";
			out << YAML::Value << static_cast<unsigned char>(distanceType);

			out << YAML::Key << "BMDistType";
			out << YAML::Value << static_cast<unsigned char>(bmdistType);

			out << YAML::Key << "weights";
			out << YAML::Value;
			out << YAML::BeginSeq;
			out << YAML::Flow << weights;
			out << YAML::EndSeq;
			out << YAML::EndMap;
			ofile << out.c_str();
			ofile.close();

			break;
		}
#if ENABLE_ROCKSDB
		case SOMFileFormat::ROCKSDB:
		{
			rocksdb::DB *db;
			rocksdb::Options options;
			// Optimize RocksDB. This is the easiest way to get RocksDB to perform well
			options.IncreaseParallelism();
			options.OptimizeLevelStyleCompaction();
			// create the DB if it's not already present
			options.create_if_missing = true;

			// open DB
			rocksdb::Status s = rocksdb::DB::Open(options, model_path, &db);
			assert(s.ok());

			s = db->Put(rocksdb::WriteOptions(), "BMDistType",
						std::to_string(static_cast<unsigned char>(bmdistType)));
			assert(s.ok());

			s = db->Put(rocksdb::WriteOptions(), "D", std::to_string(D));
			assert(s.ok());

			s = db->Put(rocksdb::WriteOptions(), "DistanceType",
						std::to_string(static_cast<unsigned char>(distanceType)));
			assert(s.ok());

			// Put H
			s = db->Put(rocksdb::WriteOptions(), "H", std::to_string(H));
			assert(s.ok());

			// Put W
			s = db->Put(rocksdb::WriteOptions(), "W", std::to_string(W));
			assert(s.ok());

			//put weights
			std::stringstream ss;
			ss << weights[0];
			for (int i = 1; i < weights.size(); ++i)
			{
				ss << ',' << weights[i];
			}
			//put weights of SOM
			s = db->Put(rocksdb::WriteOptions(), "weights", ss.str());
			assert(s.ok());
			//close db
			delete db;

			break;
		}
#endif
		default:
			break;
		}
	}
	/// <summary>
	/// get #N of columns (width) of SOM lattice
	/// </summary>
	/// <returns></returns>
	int cols() { return W; }
	/// <summary>
	/// get #N of rows (height) of SOM lattice
	/// </summary>
	/// <returns></returns>
	int rows() { return H; }
	/// <summary>
	/// get dimensions (codebook vector size) of SOM
	/// </summary>
	/// <returns></returns>
	int dims() { return D; }

	/// <summary>
	/// calculates Best Matching Unit (winning neuron).
	/// </summary>
	/// <param name="sample">input sample</param>
	/// <param name="y">index of the 0th dimension (rows) of the
	/// winning neuron </param>
	/// <param name="x">index of the 1th dimension (columns) of the
	/// winning neuron </param>
	/// <returns> distance between BMU and sample </returns>
	T calcBestMatchingUnit(const std::vector<T> &sample,
						   int &y, int &x) const
	{
		if (sample.size() != D)
		{
			throw std::runtime_error("input sample has different size than SOM");
		}

		T minDist = std::numeric_limits<T>::max();
		int min_i = 0, min_j = 0;

		switch (distanceType)
		{
		case DistanceType::Euclidean:
		{
			for (int i = 0; i < H; ++i)
			{
				for (int j = 0; j < W; ++j)
				{
					T dist = euclideanDistance(sample, nodeAt(i, j));
					if (dist < minDist)
					{
						minDist = dist;
						min_i = i;
						min_j = j;
					}
				}
			}
			break;
		}
		case DistanceType::DotProduct:
		{
			for (int i = 0; i < H; ++i)
			{
				for (int j = 0; j < W; ++j)
				{
					T dist = dotProduct(sample, nodeAt(i, j));
					//convert similarity to distance.
					dist = 1.0 / (1.0 + dist);
					if (dist < minDist)
					{
						minDist = dist;
						min_i = i;
						min_j = j;
					}
				}
			}
			break;
		}
		case DistanceType::CosineSimiarity:
		{
			for (int i = 0; i < H; ++i)
			{
				for (int j = 0; j < W; ++j)
				{
					T dist = cosineSimilarity(sample, nodeAt(i, j));
					//convert similarity to distance.
					dist = 1.0 / (1.0 + dist);
					if (dist < minDist)
					{
						minDist = dist;
						min_i = i;
						min_j = j;
					}
				}
			}
			break;
		}
		default:
		{
			break;
		}
		} //end of switch-case

		y = min_i;
		x = min_j;
		return minDist;
	} //end of the method calcBestMatchingUnit()

	friend YAML::Emitter &operator<<(YAML::Emitter &out, const SOM<T> &som)
	{
		out << YAML::BeginMap;
		out << YAML::Key << "W";
		out << YAML::Value << som.W;

		out << YAML::Key << "H";
		out << YAML::Value << som.H;

		out << YAML::Key << "D";
		out << YAML::Value << som.D;

		out << YAML::Key << "DistanceType";
		out << YAML::Value << static_cast<unsigned char>(som.distanceType);

		out << YAML::Key << "BMDistType";
		out << YAML::Value << static_cast<unsigned char>(som.bmdistType);

		out << YAML::Key << "weights";
		out << YAML::Value;
		out << YAML::Flow << som.weights;
		out << YAML::EndMap;

		return out;
	}
	friend void operator>>(const YAML::Node &node, SOM<T> &som)
	{
		som.W = node["W"].as<int>();
		som.H = node["H"].as<int>();
		som.D = node["D"].as<int>();
		som.distanceType = static_cast<DistanceType>(node["DistanceType"].as<unsigned char>());
		som.bmdistType = static_cast<BMDistType>(node["BMDistType"].as<unsigned char>());
		YAML::Node weights = node["weights"];
		som.weights.resize(weights.size());
#pragma omp parallel for
		for (int i = 0; i < weights.size(); ++i)
		{
			som.weights[i] = weights[i].as<T>();
		}
	}

  private:
	/// <summary>
	/// calculates Euclidean Distance between 2 vectors.
	/// </summary>
	/// <param name="v1">vector 1</param>
	/// <param name="v2">vector 2</param>
	/// <returns>euclidean distance</returns>
	inline T euclideanDistance(const std::vector<T> &v1,
							   const std::vector<T> &v2) const
	{
		T sum = static_cast<T>(0.0);
		for (int i = 0; i < v1.size(); ++i)
		{
			sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		}
		return sqrt(sum);
	}
	/// <summary>
	/// calculates squared euclidean distance between 2 vectors.
	/// </summary>
	/// <param name="v1">vector 1</param>
	/// <param name="v2">vector 2</param>
	/// <returns>euclidean distance</returns>
	inline T squaredEuclideanDistance(const std::vector<T> &v1,
									  const std::vector<T> &v2) const
	{
		T sum = static_cast<T>(0.0);
		for (int i = 0; i < v1.size(); ++i)
		{
			sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		}
		return sum;
	}
	/// <summary>
	/// calculates Gaussian function of given x.
	///
	/// \f$ f(x)=\frac{1}{(\sigma \sqrt{(2\pi)}}e^{(\frac{-(x-\mu)^2}{2\sigma^2})} \f$
	/// </summary>
	/// <param name="mean"> mean of the Gaussian distribution</param>
	/// <param name="stdDev">standarad deviation</param>
	/// <param name="x"> input value</param>
	/// <returns>Gaussian function of the given x.</returns>
	inline T calcGaussian(T mean, T stdDev, T x) const
	{
		// removed: (1.0 / (stdDev*sqrt(2 * (M_PI)))
		return (1.0) *
			   exp(-1.0 * (x - mean) * (x - mean) / (2.0 * stdDev * stdDev));
	}
	/// <summary>
	/// calculates 2D Gaussian function of given input pair (x,y).
	///
	/// \f$ f(x,y)=\frac{1}{(2\pi\sigma_x\sigma_y)}e^{(-[(x-\mu_x)^2/(2\sigma_x^2)+(y-\mu_y)^2
	/// /(2\sigma_y^2)])} \f$ .
	/// </summary>
	/// <param name="meanX"> mean value in X dimension. </param>
	/// <param name="meanY"> mean value in Y dimension. </param>
	/// <param name="sigmaX"> standarad deviation in X dimension. </param>
	/// <param name="sigmaY"> standarad deviation in Y dimension. </param>
	/// <param name="x"> input value (X dimension). </param>
	/// <param name="y"> input value (Y dimension). </param>
	/// <returns>2d gaussian function value of the given (x,y) pair.</returns>
	inline T calcGaussian2D(T meanX, T meanY,
							T sigmaX, T sigmaY, T x, T y) const
	{
		// / (sigmaX*sigmaY * 2 * (M_PI))
		return (1.0) *
			   exp(-1.0 * ((x - meanX) * (x - meanX) / (2.0 * sigmaX * sigmaX) + (y - meanY) * (y - meanY) / static_cast<T>(2.0 * sigmaY * sigmaY)));
	}
	/// <summary>
	/// calculates 2D Gaussian function of given input pair (x,y).
	/// This method uses the same sigma for X and Y dimensions.
	///
	///  \f$ f(x,y)=\frac{1}{(2\pi\sigma^2)}e^{(-[(x-\mu_x)^2+(y-\mu_y)^2]/(2\sigma^2))} \f$.
	/// </summary>
	/// <param name="meanX"> mean value in X dimension.</param>
	/// <param name="meanY"> mean value in Y dimension.</param>
	/// <param name="sigma"> standarad deviation (sigmaVector=[sigma, sigma])</param>
	/// <param name="x">input value (X dimension).</param>
	/// <param name="y">input value (Y dimension).</param>
	/// <returns>2d gaussian function value of the given (x,y) pair.</returns>
	inline T calcGaussian2D(int meanX, int meanY,
							T sigma, int x, int y) const
	{
		// / (sigma*sigma * 2 * (M_PI))
		return static_cast<T>((1.0) *
							  exp((-1.0 * ((x - meanX) * (x - meanX) + (y - meanY) * (y - meanY))) / (2.0 * sigma * sigma)));
	}

	/// <summary>
	/// calculates Euclidean Distance between 2 vectors.
	/// This method overloads @euclideanDistance()
	/// as second parameter is pointer to T for
	/// performance reasons.
	/// </summary>
	/// <param name="v1"> vector 1</param>
	/// <param name="v2"> vector 2</param>
	/// <returns>euclidean distance</returns>
	inline T euclideanDistance(const std::vector<T> &v1,
							   const T *v2) const
	{
		T sum = static_cast<T>(0.0);
		for (int i = 0; i < v1.size(); ++i)
		{
			sum += (v1[i] - v2[i]) * (v1[i] - v2[i]);
		}
		return sqrt(sum);
	}

	/// <summary>
	/// calculates Dot Product of 2 vectors.
	/// </summary>
	/// <param name="v1">vector 1</param>
	/// <param name="v2">vector 2</param>
	/// <returns>resulting scalar value of dot product </returns>
	inline T dotProduct(const std::vector<T> &v1,
						const T *v2) const
	{
		T sum = static_cast<T>(0.0);
		for (int i = 0; i < v1.size(); ++i)
		{
			sum += (v1[i]) * (v2[i]);
		}
		return (sum);
	}
	/// <summary>
	/// calculates cosine similarity of 2 vectors.
	/// </summary>
	/// <param name="v1">vector 1</param>
	/// <param name="v2">vector 2</param>
	/// <returns>resulting scalar value of cosine similarity</returns>
	inline T cosineSimilarity(const std::vector<T> &v1,
							  const T *v2) const
	{
		T sum = static_cast<T>(0.0);
		T v1EL = static_cast<T>(0.0);
		T v2EL = static_cast<T>(0.0);
		for (int i = 0; i < v1.size(); ++i)
		{
			sum += (v1[i]) * (v2[i]);

			v1EL += (v1[i]) * (v1[i]);
			v2EL += (v2[i]) * (v2[i]);
		}
		T cosine_sim = sum / sqrt(v1EL * v2EL);
		return (cosine_sim);
	}
	/// <summary>
	/// calculates L2 norm of a vector
	/// </summary>
	/// <param name="v1">input vector</param>
	/// <returns>scalar value of L2 norm.</returns>
	inline T L2norm(const std::vector<T> &v1)
	{
		T sum = static_cast<T>(0.0);
		for (int i = 0; i < v1.size(); ++i)
		{
			sum += (v1[i]) * (v1[i]);
		}
		return sqrt(sum);
	}

	/// <summary>
	/// Best Matching Unit
	///  neighbour distance update type
	/// of the SOM.
	/// </summary>
	BMDistType bmdistType;

	/// <summary>
	/// distance metric that is used when
	/// BMU is calculated see @calcBestMatchingUnit()
	/// </summary>
	DistanceType distanceType;

	/*
		/// <summary>
		/// current iteration
		/// </summary>
		unsigned int currIter;

		/// <summary>
		/// iteration limit
		/// </summary>
		unsigned int iterLimit;
		*/
	/// <summary>
	/// grid width
	/// </summary>
	int W;

	/// <summary>
	/// grid height
	/// </summary>
	int H;

	/// <summary>
	/// size of the weight vector of the each node.
	/// </summary>
	int D;

	/// <summary>
	/// weights / nodes of SOM
	/// </summary>
	std::vector<T> weights;
};
