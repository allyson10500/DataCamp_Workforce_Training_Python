class BinaryImageClassifier(nn.Module):
    def __init__(self):
        super(BinaryImageClassifier, self).__init__()
        
        # Create a convolutional layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
        # Create a fully connected layer
        self.fc = nn.Linear(16 * 32 * 32, 1)
        
        # Create an activation function
        self.sigmoid = nn.Sigmoid()

class MultiClassImageClassifier(nn.Module):
  
    # Define the init method
    def __init__(self, num_classes):
        super(MultiClassImageClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # Create a fully connected layer
        self.fc = nn.Linear(16*32*32, num_classes)
        
        # Create an activation function
        self.softmax = nn.Softmax(dim=1)

# Create a model
model = CNNModel()
print("Original model: ", model)

# Create a new convolutional layer
conv2 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

# Append the new layer to the model
model.add_module('conv2', conv2)
print("Extended model: ", model)

class BinaryImageClassification(nn.Module):
  def __init__(self):
    super(BinaryImageClassification, self).__init__()
    # Create a convolutional block
    self.conv_block = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
    )
    
  def forward(self, x):
    # Pass inputs through the convolutional block
    x = self.conv_block(x)
    return x

# Save the model
torch.save(model.state_dict(),"ModelCNN.pth")

# Create a new model
loaded_model = ManufacturingCNN()

# Load the saved model
loaded_model.load_state_dict(torch.load('ModelCNN.pth'))
print(loaded_model)

# Import resnet18 model
from torchvision.models import resnet18, ResNet18_Weights

# Initialize model with default weights
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Set model to evaluation mode
model.eval()

# Initialize the transforms
transform = weights.transforms()

# Apply preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Apply model with softmax layer
prediction = model(batch).squeeze(0).softmax(0)

# Apply argmax
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(category_name)

# Convert bbox into tensors
bbox_tensor = torch.tensor(bbox)

# Add a new batch dimension
bbox_tensor = bbox_tensor.unsqueeze(0)

# Resize the image and transform to tensor
transform = transforms.Compose([
  transforms.ToPILImage(),
  
])

# Apply transform to image
image_tensor = image.transform
print(image_tensor)

# Import draw_bounding_boxes
from torchvision.utils import draw_bounding_boxes

# Define the bounding box coordinates
bbox = ([x_min, y_min, x_max, y_max])
bbox_tensor = torch.tensor(bbox).unsqueeze(0)

# Implement draw_bounding_boxes
img_bbox = draw_bounding_boxes(image_tensor, bbox_tensor, width=3, colors="red")

# Tranform tensors to image
transform = transforms.Compose([
    transforms.ToPILImage()
])
plt.imshow(transform(img_bbox))
plt.show()

# Get model's prediction
with torch.no_grad():
    output = model(test_image)

# Extract boxes from the output
boxes = output[0]["boxes"]

# Extract scores from the output
scores = output[0]["scores"]

print(boxes, scores)

# Import nms
from torchvision.ops import nms

# Set the IoU threshold
iou_threshold = 0.5

# Apply non-max suppression
box_indices = nms(boxes, scores, iou_threshold)

# Filter boxes
filtered_boxes = boxes[box_indices]

print("Filtered Boxes:", filtered_boxes)

# Load pre-trained weights
vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)

# Extract the input dimension
input_dim = nn.Sequential(*list(vgg_model.classifier.children()))[0].in_features

# Create a backbone with convolutional layers
backbone = nn.Sequential(*list(vgg_model.features.children()))

# Print the backbone model
print(backbone)

# Create a variable with the number of classes
num_classes = 2
    
# Create a sequential block
classifier = nn.Sequential(
	# Create a linear layer with input features
	nn.Linear(input_dim, 512),
	nn.ReLU(),
	# Add the output dimension to the classifier
	nn.Linear(512, num_classes),
)

# Define the number of coordinates
num_coordinates = 4

bb = nn.Sequential(  
	# Add input and output dimensions
	nn.Linear(4, 32),
	nn.ReLU(),
	# Add the output for the last regression layer
	nn.Linear(32, 4),
)

# Import AnchorGenerator
from torchvision.models.detection.rpn import AnchorGenerator

# Configure anchor size
anchor_sizes = ((32, 64, 128),)

# Configure aspect ratio
aspect_ratios = ((0.5, 1.0, 2.0),)

# Instantiate AnchorGenerator
rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

# Import MultiScaleRoIAlign
from torchvision.ops import MultiScaleRoIAlign

# Instantiate RoI pooler
roi_pooler = MultiScaleRoIAlign(
	featmap_names=["0"],
	output_size=7,
	sampling_ratio=2)

mobilenet = torchvision.models.mobilenet_v2(weights="DEFAULT")
backbone = nn.Sequential(*list(mobilenet.features.children()))
backbone.out_channels = 1280

# Create Faster R-CNN model
model = FasterRCNN(
	backbone=backbone,
	num_classes=2,
	anchor_generator=anchor_generator,
	box_roi_pool=roi_pooler,
)

# Implement the RPN classification loss function
rpn_cls_criterion = nn.BCEWithLogitsLoss()

# Implement the RPN regression loss function
rpn_reg_criterion = nn.MSELoss()

# Implement the R-CNN classification Loss function
rcnn_cls_criterion = nn.CrossEntropyLoss()

# Implement the R-CNN regression loss function
rcnn_reg_criterion = nn.MSELoss()

# Load mask image
mask = Image.open("annotations/Egyptian_Mau_123.png")

# Transform mask to tensor
transform = transforms.Compose([transforms.ToTensor()])
mask_tensor = transform(mask)

# Create binary mask
binary_mask = torch.where(
    mask_tensor == 1/255,
    torch.tensor(1.0),
    torch.tensor(0.0),
)

# Print unique mask values
print(binary_mask.unique())

# Load image and transform to tensor
image = Image.open("images/Egyptian_Mau_123.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)

# Segment object out of the image
object_tensor = image_tensor * binary_mask

# Convert segmented object to image and display
to_pil_image = transforms.ToPILImage()
object_image = to_pil_image(object_tensor)
plt.imshow(object_image)
plt.show()

# Import maskrcnn_resnet50_fpn
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load a pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load an image and convert to a tensor
image = Image.open("two_cats.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)


# Perform inference
with torch.no_grad():
    prediction = model(image_tensor)
    print(prediction)

# Extract masks and labels from prediction
masks = prediction[0]["masks"]
labels = prediction[0]["labels"]

# Plot image with two overlaid masks
for i in range(2):
    plt.imshow(image)
    # Overlay the i-th mask on top of the image
    plt.imshow(masks[i, 0], cmap="jet",alpha=0.5,)
    plt.title(f"Object: {class_names[labels[i]]}")
    plt.show()

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Define the decoder blocks
        self.dec1 = self.conv_block(512, 256)
        self.dec2 = self.conv_block(256, 128)
        self.dec3 = self.conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

def forward(self, x):
    x1 = self.enc1(x)
    x2 = self.enc2(self.pool(x1))
    x3 = self.enc3(self.pool(x2))
    x4 = self.enc4(self.pool(x3))

    x = self.upconv3(x4)
    x = torch.cat([x, x3], dim=1)
    x = self.dec1(x)

    x = self.upconv2(x)
    x = torch.cat([x, x2], dim=1)
    x = self.dec2(x)

    # Define the last decoder block with skip connections
    x = self.upconv1(x)
    x = torch.cat([x, x1], dim=1)
    x = self.dec3(x)

    return self.out(x)

# Load model
model = UNet()
model.eval()

# Load and transform image
image = Image.open("car.jpg")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Predict segmentation mask
with torch.no_grad():
    prediction = model(image_tensor).squeeze(0)

# Display mask
plt.imshow(prediction[1, :, :])
plt.show()

# Instantiate the model
model = UNet()

# Produce semantic masks for the input image
with torch.no_grad():
    semantic_masks = model(image_tensor)

# Choose highest-probability class for each pixel
semantic_mask = torch.argmax(semantic_masks, dim=1)

# Display the mask
plt.imshow(semantic_mask.squeeze(0))
plt.axis("off")
plt.show()

# Instantiate model and produce instance masks
model = MaskRCNN()
with torch.no_grad():
    instance_masks = model(image_tensor)[0]["masks"]

# Initialize panoptic mask as semantic_mask
panoptic_mask = torch.clone(semantic_mask)

# Iterate over instance masks
instance_id = 3
for mask in instance_masks:
        panoptic_mask[mask > 0.5] = instance_id
        instance_id += 1
    
# Display panoptic mask
plt.imshow(panoptic_mask.squeeze(0))
plt.axis("off")
plt.show()

class Generator(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Generator, self).__init__()
        # Define generator block
        self.generator = nn.Sequential(
            gen_block(in_dim, 256),
            gen_block(256, 512),
            gen_block(512, 1024),
          	# Add linear layer
            nn.Linear(1024,out_dim),
            # Add activation
            nn.Sigmoid()
        )

    def forward(self, x):
      	# Pass input through generator
        return self.generator(x)

class Discriminator(nn.Module):
    def __init__(self, im_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            disc_block(im_dim, 1024),
            disc_block(1024, 512),
            # Define last discriminator block
            disc_block(512, 256),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # Define the forward method
        return self.disc(x)

class DCGenerator(nn.Module):
    def __init__(self, in_dim, kernel_size=4, stride=2):
        super(DCGenerator, self).__init__()
        self.in_dim = in_dim
        self.gen = nn.Sequential(
            dc_gen_block(in_dim, 1024, kernel_size, stride),
            dc_gen_block(1024, 512, kernel_size, stride),
            # Add last generator block
            dc_gen_block(512, 256, kernel_size, stride),
            nn.ConvTranspose2d(256, 3, kernel_size, stride=stride),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(len(x), self.in_dim, 1, 1)
        return self.gen(x)

class DCDiscriminator(nn.Module):
    def __init__(self, kernel_size=4, stride=2):
        super(DCDiscriminator, self).__init__()
        self.disc = nn.Sequential(
          	# Add first discriminator block
            dc_disc_block(3, 512, kernel_size, stride),
            dc_disc_block(512, 1024, kernel_size, stride),
          	# Add a convolution
            nn.Conv2d(1024, 1, kernel_size, stride=stride),
        )

    def forward(self, x):
        # Pass input through sequential block
        x = x.view(len(x), self.in_dim, 1, 1)
        return x.view(len(x), -1)

def gen_loss(gen, disc, criterion, num_images, z_dim):
    # Define random noise
    noise = torch.randn(num_images, z_dim)
    # Generate fake image
    fake = gen(noise)
    # Get discriminator's prediction on the fake image
    disc_pred = disc(fake)
    # Compute generator loss
    criterion = nn.BCEWithLogitsLoss()
    gen_loss = criterion(disc_pred, torch.ones_like(disc_pred))
    return gen_loss

def disc_loss(gen, disc, real, num_images, z_dim):
    criterion = nn.BCEWithLogitsLoss()
    noise = torch.randn(num_images, z_dim)
    fake = gen(noise)
    # Get discriminator's predictions for fake images
    disc_pred_fake = disc(fake)
    # Calculate the fake loss component
    fake_loss = criterion(disc_pred_fake,torch.zeros_like(disc_pred_fake))
    # Get discriminator's predictions for real images
    disc_pred_real = disc(real)
    # Calculate the real loss component
    real_loss = criterion(disc_pred_real,torch.ones_like(disc_pred_real))
    disc_loss = (real_loss + fake_loss) / 2
    return disc_loss

for epoch in range(1):
    for real in dataloader:
        cur_batch_size = len(real)
        
        disc_opt.zero_grad()
        # Calculate discriminator loss
        disc_loss = disc_loss(gen, disc, real, cur_batch_size, z_dim=16)
        # Compute gradients
        disc_loss.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        # Calculate generator loss
        gen_loss = gen_loss(gen, disc, cur_batch_size, z_dim=16)
        # Compute generator gradients
        gen_loss.backward()

        gen_opt.step()

        print(f"Generator loss: {gen_loss}")
        print(f"Discriminator loss: {disc_loss}")
        break

num_images_to_generate = 5
# Create random noise tensor
noise = torch.randn(num_images_to_generate, 16)
# Generate images
with torch.no_grad():
    fake = gen(noise)
print(f"Generated tensor shape: {fake.shape}")
    
for i in range(num_images_to_generate):
    # Slice fake to select i-th image
    image_tensor = fake[i, :, :, :]
    # Permute the image dimensions
    image_tensor_permuted = image_tensor.permute(1, 2, 0)
    plt.imshow(image_tensor_permuted)
    plt.show()

# Import FrechetInceptionDistance
from torchmetrics.image.fid import FrechetInceptionDistance

# Instantiate FID
fid = FrechetInceptionDistance(feature=64)

# Update FID with real images
fid.update((fake * 255).to(torch.uint8), real=False)
fid.update((real * 255).to(torch.uint8), real=True)

# Compute the metric
fid_score = fid.compute()
print(fid_score)