import streamlit as st
import torchvision


st.title("PneumonIA")

with st.sidebar:
        st.write("Hello our names are Zoé DUPRAT, Martin CORNEN, Camille GRISLIN.") 
        st.write("We decided to create an app where you can drag your chest x-ray and an IA will determine your pourcentage of chance to have a pneumonia.")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    st.write(uploaded_file.name)

def img_to_torch(pil_image):
	img = pil_image.convert('L')
	x = torchvision.transforms.functional.to_tensor(img)
	x = torchvision.transforms.functional.resize(x, [150, 150])
	x.unsqueeze_(0)
	return x

def predict(image, model):
	x = img_to_torch(image)
	pred = model(x)
	pred = pred.detach().numpy()

	df = pd.DataFrame(data=pred[0], index=['Bacterial', 'Normal', 'Viral'], columns=['confidence'])

	st.write(f'''### 🧫 Confidence - Bacterial:  **{np.round(pred[0][0]*100, 3)}%**''')
	st.write(f'''### 🦠 Confidence - Viral: **{np.round(pred[0][2]*100, 3)}%**''')
	st.write(f'''### 👍 Confidence - Normal: **{np.round(pred[0][1]*100, 3)}%**''')
	st.write('')
	st.bar_chart(df)

PATH_TO_MODEL = './sm_91.pt'
model = torch.load(PATH_TO_MODEL)
model.eval()

uploaded_file = st.file_uploader('Upload image...', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='This x-ray will be diagnosed...', use_column_width=True)

	if st.button('Predict 🧠'):
		predict(image, model)
