# Demos and Tutorials

This section lists several demos that apply Concrete ML to some popular machine learning problems. They show
how to build ML models that perform well under FHE constraints, and then how to perform the conversion to FHE.

Simpler tutorials that discuss only model usage and compilation are also available for [built-in models](../built-in-models/ml_examples.md) and [deep learning](../deep-learning/examples.md).

<table data-view="cards">
   <thead>
      <tr>
         <th></th>
         <th></th>
         <th></th>
         <th data-hidden data-card-cover data-type="files"></th>
         <th data-hidden data-card-target data-type="content-ref"></th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td><strong>GPT-2 in FHE</strong></td>
         <td>
            <p></p>
            <p>Privacy-preserving text generation based on a user's prompt</p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_gpt2.png">health.png</a></td>
         <td><a href="../../use_case_examples/llm">use_case_examples/llm</a></td>
         <!--- end -->
      </tr>        
      <tr>
         <td><strong>Titanic</strong></td>
         <td>
            <p></p>
            <p>Train an XGB classifier that can perform encrypted prediction for the <a href="https://www.kaggle.com/c/titanic/">Kaggle Titanic competition</a></p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_titanic.png">titanic.png</a></td>
         <td><a href="../../use_case_examples/titanic">use_case_examples/titanic</a></td>
         <!--- end -->
      </tr>
      <tr>
         <td><strong>Neural Network Fine-tuning</strong> </td>
         <td>
            <p></p>
            <p>Fine-tune a VGG network to classify the CIFAR image data-sets and predict on encrypted data</p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_nn_finetuning.png">nn.png</a></td>
         <td><a href="../../use_case_examples/cifar/cifar_brevitas_finetuning">use_case_examples/cifar/cifar_brevitas_finetuning</a></td>
         <!--- end -->
      </tr>
      <tr>
         <td><strong>Neural Network Splitting for SaaS deployment</strong> </td>
         <td>
            <p></p>
            <p>Train a VGG-like CNN that classifies CIFAR10 encrypted images, and where an initial feature extractor is executed client-side</p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_nn_splitting.png">client-server-1.png</a></td>
         <td><a href="../../use_case_examples/cifar/cifar_brevitas_with_model_splitting">use_case_examples/cifar/cifar_brevitas_with_model_splitting</a></td>
         <!--- end -->
      </tr>
      <tr>
         <td><strong>Encrypted Image filtering</strong></td>
         <td>
            <p></p>
            <p>A Hugging Face space that applies a variety of image filters to encrypted images</p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_filtering.png">blurring.png</a></td>
         <td><a href="../../use_case_examples/image_filtering">use_case_examples/image_filtering</a></td>
         <!--- end -->
      </tr>
      <tr>
         <td><strong>Encrypted sentiment analysis</strong></td>
         <td>
            <p></p>
            <p>A Hugging Face space that securely analyzes the sentiment expressed in a short text</p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_sentiment.png">sentiment.png</a></td>
         <td><a href="../../use_case_examples/sentiment_analysis_with_transformer">use_case_examples/sentiment_analysis_with_transformer</a></td>
         <!--- end -->
      </tr>
      <tr>
         <td><strong>Credit Scoring</strong></td>
         <td>
            <p></p>
            <p>Predict the chance of a given loan applicant defaulting on loan repayment </p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_credit.png">credit.png</a></td>
         <td><a href="../../use_case_examples/credit_scoring">use_case_examples/credit_scoring</a></td>
         <!--- end -->
      </tr>
      <tr>
         <td><strong>Healthcare diagnosis</strong></td>
         <td>
            <p></p>
            <p>Give a diagnosis using FHE to preserve the privacy of the patient</p>
         </td>
         <td></td>
         <!--- start -->
         <td><a href="../.gitbook/assets/demo_health.png">health.png</a></td>
         <td><a href="../../use_case_examples/disease_prediction">use_case_examples/disease_prediction</a></td>
         <!--- end -->
      </tr>      
   </tbody>
</table>
