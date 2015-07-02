#ifndef APPLYALGOFILTER_H
#define APPLYALGOFILTER_H

#include <Eigen/Dense>
#include <unsupported/Eigen/LevenbergMarquardt>
#include <unsupported/Eigen/NumericalDiff>

#include "../Sequence.h"
#include "../Model.h"

#include "itkImageToImageFilter.h"
#include "itkSliceBySliceImageFilter.h"

template<typename DataType>
class Algorithm {
public:
	typedef DataType TScalar;
	typedef Eigen::Array<TScalar, Eigen::Dynamic, 1> TInput;
	typedef Eigen::ArrayXd TArray;
	virtual size_t numInputs() const = 0;  // The number of inputs that will be concatenated into the data vector
	virtual size_t numConsts() const = 0;  // Number of constant input parameters/variables
	virtual size_t numOutputs() const = 0; // Number of output parameters/variables
	virtual size_t dataSize() const = 0;   // The expected size of the concatenated data vector

	virtual void apply(const TInput &data,
					   const TArray &consts,
					   TArray &outputs,
					   TArray &resids) const = 0;

	virtual TArray defaultConsts() = 0;
};

namespace itk{

template<typename TAlgorithm , typename TData, typename TScalar, unsigned int ImageDim>
class ApplyAlgorithmFilter :
	public ImageToImageFilter<VectorImage<TData, ImageDim>, VectorImage<TScalar, ImageDim>>
{
public:
	typedef VectorImage<TData, ImageDim>   TDataVectorImage;
	typedef VectorImage<TScalar, ImageDim> TScalarVectorImage;
	typedef Image<TScalar, ImageDim>       TScalarImage;

	typedef ApplyAlgorithmFilter                                     Self;
	typedef ImageToImageFilter<TDataVectorImage, TScalarVectorImage> Superclass;
	typedef SmartPointer<Self>                                       Pointer;
	typedef typename TScalarImage::RegionType                        TRegion;

	itkNewMacro(Self); /** Method for creation through the object factory. */
	itkTypeMacro(ApplyAlgorithmFilter, ImageToImageFilter); /** Run-time type information (and related methods). */

	void SetAlgorithm(const shared_ptr<TAlgorithm> &a);
	shared_ptr<const TAlgorithm> GetAlgorithm() const;
	void SetScaleToMean(const bool s);
	bool GetScaleToMean() const;

	void SetInput(const size_t i, const TDataVectorImage *img);
	typename TDataVectorImage::ConstPointer GetInput(const size_t i) const;
	void SetConst(const size_t i, const TScalarImage *img);
	typename TScalarImage::ConstPointer GetConst(const size_t i) const;
	void SetMask(const TScalarImage *mask);
	typename TScalarImage::ConstPointer GetMask() const;

	TScalarImage *GetOutput(const size_t i);
	TScalarVectorImage *GetResidOutput();

	virtual void GenerateOutputInformation() override;

protected:
	ApplyAlgorithmFilter();
	~ApplyAlgorithmFilter(){}

	virtual void ThreadedGenerateData(const TRegion &outputRegionForThread, ThreadIdType threadId) override;
	DataObject::Pointer MakeOutput(unsigned int idx);

	shared_ptr<TAlgorithm> m_algorithm;
	bool m_scale_to_mean = false;

private:
	ApplyAlgorithmFilter(const Self &); //purposely not implemented
	void operator=(const Self &);  //purposely not implemented
};
}

#include "ApplyAlgorithmFilter.hxx"

#endif // APPLYAGLOFILTER_H
