#define _USE_MATH_DEFINES
#include <cmath> 
#include <iostream>
#include <cstdlib>
#include <chronoCPU.hpp>
#include "ExerciseImpl.h"
#include <random>
#include <ppm.h>
#include <reference/ImageEqualizer.h>


namespace{

};

// ==========================================================================================
void ExerciseImpl::usage( const char*const prg ) {
    #ifdef WIN32
    const char*last_slash = strrchr(prg, '\\');
    #else
    const char*last_slash = strrchr(prg, '/');
    #endif
    std::cout << "Usage: " << (last_slash==nullptr ? prg : last_slash+1) 
        << " [ -i=<input.ppm> ]" << std::endl 
        << "where -i  specifies the file name of the input image."
        << std::endl;
}

// ==========================================================================================
void ExerciseImpl::usageAndExit( const char*const prg, const int code ) {
    usage(prg);
    exit( code );
}

// ==========================================================================================
void ExerciseImpl::displayHelpIfNeeded(const int argc, const char**argv) 
{
    if( checkCmdLineFlag(argc, argv, "-h") || checkCmdLineFlag(argc, argv, "help") ) {
        usageAndExit(argv[0], EXIT_SUCCESS);
    }
}
ExerciseImpl& ExerciseImpl::parseCommandLine(const int argc, const char**argv) 
{
    displayHelpIfNeeded(argc, argv);
    std::cout << "argv[1] = " << argv[1] << std::endl;
    if( checkCmdLineFlag(argc, argv, "i") ) {
        if( !getCmdLineArgumentString(argc, argv, "i", &inputFileName) ) 
        {
            std::cerr << "unable to get the input file name" << std::endl;
            usageAndExit(argv[0], -1);  
        }        
    }
    if( inputFileName == nullptr )
        usageAndExit(argv[0], -1);
    return *this;
}

void ExerciseImpl::prepare_data() 
{
    std::cout << "- load input file " << inputFileName << std::endl;
    sourceImage = new PPMBitmap(inputFileName);
    const unsigned width = sourceImage->getWidth();
    const unsigned height = sourceImage->getHeight();
    std::cout << "- File <" << inputFileName << "> loaded. Contains "<<width<< " per " << height << " pixels." << std::endl;
    destImage = new PPMBitmap(width, height);
    Transformation.resize(width*height);
    prepare_truth();
}



void ExerciseImpl::run(const bool verbose) {    
    prepare_data();
    const unsigned nbTry = 10u;
    if( verbose ) {
        std::cout << "Student code will run " << nbTry << " times for statistics ..." << std::endl;
    }

    const unsigned width = sourceImage->getWidth();
    const unsigned height = sourceImage->getHeight();
    const unsigned size = width * height;
	OPP::CUDA::DeviceBuffer<uchar3> dev_input(reinterpret_cast<uchar3*>(sourceImage->getPtr()), size);
	OPP::CUDA::DeviceBuffer<float> dev_Value(reference->getValue().data(), size);
	OPP::CUDA::DeviceBuffer<unsigned> dev_Repartition(reference->getRepartition().data(), 256);
	OPP::CUDA::DeviceBuffer<float> dev_Transformation(size);
	
    if( verbose )
        std::cout << " - compute the transformation function " << std::endl;
    execute_and_display_GPU_time(verbose, [&]() {
        auto impl = reinterpret_cast<StudentWorkImpl*>(student);
        impl->run_Transformation(dev_Value, dev_Repartition, dev_Transformation);
    }, nbTry);

	dev_Transformation.copyToHost(Transformation.data());	
}


bool ExerciseImpl::check() 
{
    saveTo(makeFileName(inputFileName, "_equalized.ppm"), *destImage);
    if( !check_Transformation() ) return false;
    return true;
}

std::string ExerciseImpl::makeFileName(const char*fileName, const std::string& extension)
{
    std::string s_output(fileName);
    size_t delimiter_pos = s_output.find_last_of('.');
    s_output.resize(delimiter_pos);
    s_output.append(extension);
    return s_output;
}
    
void ExerciseImpl::saveTo(const std::string& fileName, PPMBitmap& image)
{
    std::cout << "Save result into " << fileName << std::endl;
    image.saveTo(fileName.c_str());
}


bool ExerciseImpl::checkImagesAreEquals(const PPMBitmap&imageA, const PPMBitmap&imageB)
{
    if( imageA.getWidth() != imageB.getWidth() || imageA.getHeight() != imageB.getHeight() )
        return false;
    const uchar* ptrA = imageA.getPtr();
    const uchar* ptrB = imageB.getPtr();
    for(unsigned i=imageA.getWidth()*imageA.getWidth(); i--;)
        if( std::max(ptrA[i],ptrB[i]) - std::min(ptrA[i],ptrB[i]) > 1 ) return false;
    return true;
}
    

bool ExerciseImpl::check_Transformation()
{
    for(int i=0; i<64; ++i) std::cout << Transformation[i] << " ";
    std::cout << std::endl;
    for(int i=0; i<64; ++i) std::cout << reference->getTransformation()[i] << " ";
    std::cout << std::endl;
    const auto& refTransformation = reference->getTransformation();
    for(auto i=Transformation.size(); i--;)
        if( std::abs(Transformation[i] - refTransformation[i] ) >= 1.f/256.f ) {
            std::cout << "Results differ!" << std::endl;
            return false;
        }
    return true;
}


void ExerciseImpl::prepare_truth() 
{
    std::cout << "Calculate the reference Histogram on CPU ..." << std::endl;
    reference = new ImageEqualizer(*sourceImage, *destImage); 
    reference->computeHSVfromRGB();   
    reference->computeHistogramFromValue();
    reference->computeRepartitionFunction();
    const unsigned nbTry = 10u;
    execute_and_display_time(
        true, 
        [&]() { reference->computeFinalTransformation(); }, 
        nbTry,
        std::string("\tDone in")
    );    
    reference->applyFinalTransformation();
}